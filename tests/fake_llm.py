"""Modelo de chat falso con respuestas guionadas, para probar sin API key.

Implementa ``bind_tools`` para que ``with_structured_output`` funcione igual
que con un proveedor real (tool calling + ``PydanticToolsParser``). Cada
llamada devuelve la siguiente respuesta del guion, lo que permite simular
JSON mal formado, objetos incompletos o respuestas truncadas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

NOMBRE_TOOL = "ExtraccionTecnica"


class ModeloGuionado(BaseChatModel):
    """Devuelve, en orden, los mensajes de ``respuestas`` (repite el último)."""

    respuestas: list[AIMessage]
    llamadas: int = 0
    prompts_recibidos: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "modelo-guionado"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ModeloGuionado:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        indice = min(self.llamadas, len(self.respuestas) - 1)
        self.llamadas += 1
        self.prompts_recibidos.append(list(messages))
        mensaje = self.respuestas[indice].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def respuesta_valida(
    tecnologias: list[str] | None = None,
    nivel: str = "alta",
    resumen: str = "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes.",
    finish_reason: str = "stop",
) -> AIMessage:
    """Tool call con argumentos que cumplen el esquema."""
    return respuesta_con_args(
        {
            "tecnologias": tecnologias or ["FastAPI", "Redis", "PostgreSQL"],
            "nivel_de_criticidad": nivel,
            "resumen_tecnico": resumen,
        },
        finish_reason=finish_reason,
    )


def respuesta_con_args(args: dict[str, Any], finish_reason: str = "stop") -> AIMessage:
    """Tool call con argumentos arbitrarios (pueden violar el esquema)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": NOMBRE_TOOL, "args": args, "id": "call_1", "type": "tool_call"}
        ],
        response_metadata={"finish_reason": finish_reason},
    )


def respuesta_json_mal_formado(finish_reason: str = "stop") -> AIMessage:
    """Tool call cuyo JSON no se pudo parsear (llega como ``invalid_tool_calls``)."""
    return AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": NOMBRE_TOOL,
                "args": '{"tecnologias": ["FastAPI", "Redis"], "nivel_de_criticidad": "al',
                "id": "call_1",
                "error": "JSONDecodeError: Unterminated string",
                "type": "invalid_tool_call",
            }
        ],
        response_metadata={"finish_reason": finish_reason},
    )


def respuesta_solo_texto(contenido: str = "No puedo identificar tecnologías.") -> AIMessage:
    """El modelo respondió en texto libre en lugar de llamar a la tool."""
    return AIMessage(content=contenido, response_metadata={"finish_reason": "stop"})
