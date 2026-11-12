"""Mini-script de prueba asíncrono del pipeline.

Procesa en paralelo (``asyncio.gather``) tres textos: una descripción de
arquitectura, un log de error y un texto ambiguo (prueba de estrés).

Uso:
    python prueba_async.py                  # usa el LLM configurado en .env
    python prueba_async.py --max-tokens 30  # fuerza respuestas truncadas
    python prueba_async.py --offline        # sin API key: modelo falso con fallas guionadas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from langchain_core.runnables import Runnable

from chain import ExtraccionFallidaError, build_chain, build_llm, process_text
from schemas import ExtraccionTecnica

EJEMPLOS: dict[str, str] = {
    "arquitectura": (
        "Nuestra API está construida con FastAPI y corre detrás de Nginx. Usamos Redis "
        "como caché de sesiones y PostgreSQL como base principal. En horas pico el pool "
        "de conexiones de PostgreSQL se agota con más de 500 requests concurrentes y la "
        "latencia p99 sube a 4 segundos, lo que provoca timeouts en el frontend."
    ),
    "log_error": (
        "2026-11-12T03:14:07Z ERROR worker-7 celery.app.trace Task "
        "billing.charge_invoice[9f2c] raised unexpected: OperationalError: "
        "(psycopg2.OperationalError) FATAL: remaining connection slots are reserved "
        "for non-replication superuser connections\n"
        "kubelet: Back-off restarting failed container billing-worker in pod "
        "billing-worker-6d9f (CrashLoopBackOff)\n"
        "RabbitMQ queue 'billing' messages_ready=48213"
    ),
    "ambiguo": (
        "Ayer volvió a pasar lo de siempre: algo se puso lento, la gente se quejó y "
        "después se arregló solo. Nadie sabe bien qué fue."
    ),
}


def configurar_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx registra cada request HTTP en INFO; se silencia para que se lean
    # los logs de validación y reintentos.
    for ruidoso in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def cadenas_offline() -> dict[str, Runnable[dict[str, str], ExtraccionTecnica]]:
    """Una cadena por ejemplo, cada una con un modelo falso que falla a propósito."""
    from tests.fake_llm import (
        ModeloGuionado,
        respuesta_con_args,
        respuesta_json_mal_formado,
        respuesta_valida,
    )

    guiones = {
        # 1.º JSON mal formado -> reintento -> objeto válido.
        "arquitectura": [
            respuesta_json_mal_formado(),
            respuesta_valida(
                tecnologias=["FastAPI", "Nginx", "Redis", "PostgreSQL"],
                nivel="alta",
                resumen=(
                    "API FastAPI tras Nginx con caché en Redis y PostgreSQL; el pool de "
                    "conexiones se agota en horas pico y dispara timeouts."
                ),
            ),
        ],
        # 1.º respuesta cortada por tokens -> reintento -> objeto válido.
        "log_error": [
            respuesta_valida(finish_reason="length"),
            respuesta_valida(
                tecnologias=["Celery", "PostgreSQL", "Kubernetes", "RabbitMQ"],
                nivel="alta",
                resumen=(
                    "Workers de Celery agotan las conexiones de PostgreSQL, el pod entra "
                    "en CrashLoopBackOff y la cola de RabbitMQ acumula mensajes."
                ),
            ),
        ],
        # Siempre devuelve una lista de tecnologías vacía -> se agotan los reintentos.
        "ambiguo": [
            respuesta_con_args(
                {
                    "tecnologias": [],
                    "nivel_de_criticidad": "media",
                    "resumen_tecnico": "Lentitud intermitente sin causa identificada.",
                }
            )
        ],
    }
    return {
        nombre: build_chain(ModeloGuionado(respuestas=guion), esperar_entre_intentos=False)
        for nombre, guion in guiones.items()
    }


async def ejecutar(
    nombre: str, texto: str, cadena: Runnable[dict[str, str], ExtraccionTecnica]
) -> tuple[str, ExtraccionTecnica | None, str | None]:
    try:
        resultado = await process_text(texto, chain=cadena, etiqueta=nombre)
        return nombre, resultado, None
    except ExtraccionFallidaError as exc:
        return nombre, None, str(exc)


async def main(offline: bool, max_tokens: int | None) -> int:
    if offline:
        cadenas = cadenas_offline()
    else:
        cadena = build_chain(build_llm(max_tokens=max_tokens))
        cadenas = {nombre: cadena for nombre in EJEMPLOS}

    resultados = await asyncio.gather(
        *(ejecutar(nombre, texto, cadenas[nombre]) for nombre, texto in EJEMPLOS.items())
    )

    print("\n" + "=" * 72)
    exitosos = 0
    for nombre, resultado, error in resultados:
        print("\n### " + nombre)
        if resultado is not None:
            exitosos += 1
            print(json.dumps(resultado.model_dump(mode="json"), indent=2, ensure_ascii=False))
        else:
            print("FALLÓ LA VALIDACIÓN -> " + (error or ""))
    print("\n" + "=" * 72)
    print("Resumen: " + str(exitosos) + "/" + str(len(resultados)) + " textos con objeto válido")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="usar un modelo falso (no requiere API key)")
    parser.add_argument("--max-tokens", type=int, default=None, help="límite de tokens de salida del LLM")
    argumentos = parser.parse_args()

    configurar_logs()
    sys.exit(asyncio.run(main(argumentos.offline, argumentos.max_tokens)))
