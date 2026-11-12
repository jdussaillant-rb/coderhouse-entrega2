import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelCriticidad

EJEMPLO_ENUNCIADO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": (
        "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en "
        "conexiones concurrentes."
    ),
}


def test_acepta_el_ejemplo_del_enunciado():
    resultado = ExtraccionTecnica.model_validate(EJEMPLO_ENUNCIADO)

    assert resultado.model_dump(mode="json") == EJEMPLO_ENUNCIADO


def test_criticidad_solo_acepta_baja_media_alta():
    assert {n.value for n in NivelCriticidad} == {"baja", "media", "alta"}
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**EJEMPLO_ENUNCIADO, "nivel_de_criticidad": "critica"})


def test_rechaza_lista_de_tecnologias_vacia():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**EJEMPLO_ENUNCIADO, "tecnologias": []})


def test_rechaza_lista_que_solo_tiene_blancos():
    with pytest.raises(ValidationError, match="no puede quedar vacía"):
        ExtraccionTecnica.model_validate({**EJEMPLO_ENUNCIADO, "tecnologias": ["  ", ""]})


def test_limpia_espacios_y_duplicados_de_tecnologias():
    resultado = ExtraccionTecnica.model_validate(
        {**EJEMPLO_ENUNCIADO, "tecnologias": [" Redis ", "redis", "PostgreSQL", ""]}
    )

    assert resultado.tecnologias == ["Redis", "PostgreSQL"]


@pytest.mark.parametrize("resumen", ["", "corto", "x" * 401])
def test_rechaza_resumen_fuera_de_rango(resumen):
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**EJEMPLO_ENUNCIADO, "resumen_tecnico": resumen})


@pytest.mark.parametrize("campo", ["tecnologias", "nivel_de_criticidad", "resumen_tecnico"])
def test_rechaza_objeto_incompleto(campo):
    incompleto = {k: v for k, v in EJEMPLO_ENUNCIADO.items() if k != campo}

    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(incompleto)


def test_rechaza_campos_extra():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**EJEMPLO_ENUNCIADO, "inventado": 1})
