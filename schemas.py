"""Contrato de salida del pipeline de extracción de entidades técnicas.

Este módulo define el esquema Pydantic que el LLM debe respetar. Las
restricciones declaradas aquí se validan en el cliente: si el modelo devuelve
un objeto que no las cumple, la cadena lo trata como respuesta inválida y
reintenta (ver ``chain.py``).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TECNOLOGIAS = 20
MIN_LARGO_RESUMEN = 15
MAX_LARGO_RESUMEN = 400


class NivelCriticidad(str, Enum):
    """Severidad del problema o riesgo descrito en el texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log, incidente o arquitectura)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
        extra="forbid",
    )

    tecnologias: list[str] = Field(
        min_length=1,
        max_length=MAX_TECNOLOGIAS,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos, servicios o "
            "herramientas mencionadas o claramente implicadas en el texto, con su "
            "nombre canónico (ej. 'PostgreSQL', no 'postgres'). Al menos una."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Severidad del problema descrito: 'alta' si hay caída, pérdida de "
            "datos, riesgo de seguridad o impacto en producción; 'media' si hay "
            "degradación o errores intermitentes; 'baja' si es informativo o sin "
            "impacto inmediato."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=MIN_LARGO_RESUMEN,
        max_length=MAX_LARGO_RESUMEN,
        description=(
            "Resumen técnico en español, de una o dos oraciones, que explique la "
            "arquitectura o el problema y su causa probable."
        ),
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Descarta entradas vacías y duplicados (sin distinguir mayúsculas)."""
        vistas: set[str] = set()
        limpias: list[str] = []
        for valor in valores:
            nombre = valor.strip()
            clave = nombre.casefold()
            if nombre and clave not in vistas:
                vistas.add(clave)
                limpias.append(nombre)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpias
