import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from chain import (
    MAX_INTENTOS,
    ExtraccionFallidaError,
    build_chain,
    build_llm,
    build_prompt,
    process_text,
)
from schemas import ExtraccionTecnica
from tests.fake_llm import (
    ModeloGuionado,
    respuesta_con_args,
    respuesta_json_mal_formado,
    respuesta_solo_texto,
    respuesta_valida,
)

TEXTO = "La API en FastAPI usa Redis y PostgreSQL; se satura con conexiones concurrentes."


def cadena_con(*respuestas: AIMessage) -> tuple[ModeloGuionado, object]:
    modelo = ModeloGuionado(respuestas=list(respuestas))
    return modelo, build_chain(modelo, esperar_entre_intentos=False)


# --- Prompt -------------------------------------------------------------------


def test_prompt_recibe_texto_e_instrucciones_de_formato():
    prompt = build_prompt()

    assert prompt.input_variables == ["texto"]
    assert "instrucciones_formato" in prompt.partial_variables

    mensajes = prompt.format_messages(texto="log con {llaves} y Redis")
    sistema, humano = mensajes[0].content, mensajes[1].content
    for campo in ExtraccionTecnica.model_fields:
        assert campo in sistema
    assert "log con {llaves} y Redis" in humano


# --- Camino feliz ------------------------------------------------------------


async def test_devuelve_objeto_validado_en_el_primer_intento():
    modelo, cadena = cadena_con(respuesta_valida())

    resultado = await process_text(TEXTO, chain=cadena)

    assert isinstance(resultado, ExtraccionTecnica)
    assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert resultado.nivel_de_criticidad == "alta"
    assert modelo.llamadas == 1


async def test_el_texto_llega_al_llm_dentro_del_prompt():
    modelo, cadena = cadena_con(respuesta_valida())

    await process_text(TEXTO, chain=cadena)

    assert TEXTO in modelo.prompts_recibidos[0][-1].content


# --- Resiliencia: el modelo se recupera ---------------------------------------


@pytest.mark.parametrize(
    "respuesta_fallida",
    [
        pytest.param(respuesta_json_mal_formado(), id="json-mal-formado"),
        pytest.param(respuesta_con_args({"tecnologias": ["Redis"]}), id="objeto-incompleto"),
        pytest.param(
            respuesta_con_args(
                {"tecnologias": [], "nivel_de_criticidad": "alta", "resumen_tecnico": "x" * 30}
            ),
            id="lista-vacia",
        ),
        pytest.param(
            respuesta_con_args(
                {
                    "tecnologias": ["Redis"],
                    "nivel_de_criticidad": "urgente",
                    "resumen_tecnico": "x" * 30,
                }
            ),
            id="enum-invalido",
        ),
        pytest.param(respuesta_solo_texto(), id="sin-tool-call"),
        pytest.param(respuesta_valida(finish_reason="length"), id="truncado-openai"),
    ],
)
async def test_reintenta_y_se_recupera(respuesta_fallida):
    modelo, cadena = cadena_con(respuesta_fallida, respuesta_valida())

    resultado = await process_text(TEXTO, chain=cadena)

    assert isinstance(resultado, ExtraccionTecnica)
    assert modelo.llamadas == 2


async def test_detecta_truncado_de_anthropic_por_stop_reason():
    truncado = respuesta_valida()
    truncado.response_metadata = {"stop_reason": "max_tokens"}
    modelo, cadena = cadena_con(truncado, respuesta_valida())

    await process_text(TEXTO, chain=cadena)

    assert modelo.llamadas == 2


async def test_un_objeto_truncado_se_descarta_aunque_sea_parseable():
    """Un objeto aparentemente válido pero cortado por tokens no se acepta."""
    modelo, cadena = cadena_con(
        respuesta_valida(finish_reason="length"),
        respuesta_valida(finish_reason="length"),
        respuesta_valida(finish_reason="length"),
    )

    with pytest.raises(ExtraccionFallidaError, match="incompleta"):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == MAX_INTENTOS


# --- Resiliencia: se agotan los reintentos ------------------------------------


async def test_lanza_error_al_agotar_los_reintentos():
    modelo, cadena = cadena_con(respuesta_json_mal_formado())

    with pytest.raises(ExtraccionFallidaError, match=str(MAX_INTENTOS) + " intento"):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == MAX_INTENTOS


async def test_no_reintenta_si_el_modelo_se_niega():
    rechazo = respuesta_solo_texto("No puedo ayudar con eso.")
    rechazo.response_metadata = {"stop_reason": "refusal"}
    modelo, cadena = cadena_con(rechazo, respuesta_valida())

    with pytest.raises(ExtraccionFallidaError, match="negó"):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 1


# --- Entrada y logs -----------------------------------------------------------


@pytest.mark.parametrize("texto", ["", "   \n", "x" * 8_001])
async def test_rechaza_entrada_invalida_sin_llamar_al_llm(texto):
    modelo, cadena = cadena_con(respuesta_valida())

    with pytest.raises(ValueError):
        await process_text(texto, chain=cadena)
    assert modelo.llamadas == 0


async def test_registra_validacion_y_reintentos(caplog):
    _, cadena = cadena_con(respuesta_json_mal_formado(), respuesta_valida())

    with caplog.at_level(logging.INFO, logger="pipeline"):
        await process_text(TEXTO, chain=cadena, etiqueta="caso-log")

    mensajes = [r.getMessage() for r in caplog.records]
    assert any("[caso-log] Intento 1" in m for m in mensajes)
    assert any("[caso-log] El LLM no devolvió un objeto estructurado" in m for m in mensajes)
    assert any("[caso-log] Reintento automático: intento 2" in m for m in mensajes)
    assert any("[caso-log] Validación OK" in m for m in mensajes)
    assert any("[caso-log] Extracción válida en 2 intento(s)" in m for m in mensajes)


async def test_procesa_varios_textos_en_paralelo():
    _, cadena_a = cadena_con(respuesta_valida(nivel="baja"))
    _, cadena_b = cadena_con(respuesta_json_mal_formado(), respuesta_valida(nivel="media"))

    a, b = await asyncio.gather(
        process_text(TEXTO, chain=cadena_a, etiqueta="a"),
        process_text(TEXTO, chain=cadena_b, etiqueta="b"),
    )

    assert (a.nivel_de_criticidad, b.nivel_de_criticidad) == ("baja", "media")


# --- Configuración del cliente -----------------------------------------------


@pytest.fixture
def sin_dotenv(monkeypatch):
    monkeypatch.setattr("chain.load_dotenv", lambda: None)


@pytest.mark.parametrize("proveedor, variable", [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
def test_build_llm_exige_la_clave_del_proveedor(monkeypatch, sin_dotenv, proveedor, variable):
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(RuntimeError, match=variable):
        build_llm(proveedor)


def test_build_llm_rechaza_proveedor_desconocido(sin_dotenv):
    with pytest.raises(ValueError, match="no soportado"):
        build_llm("gemini")


@pytest.mark.parametrize("proveedor", ["openai", "anthropic"])
def test_build_llm_crea_cliente_real_con_limite_de_tokens(monkeypatch, sin_dotenv, proveedor):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    llm = build_llm(proveedor, max_tokens=77)

    assert llm.max_tokens == 77
    build_chain(llm)  # with_structured_output acepta el esquema sin llamar a la API
