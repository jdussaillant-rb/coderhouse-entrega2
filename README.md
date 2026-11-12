# Pre-entrega 2: Pipeline de extracción de entidades técnicas

Pipeline en LangChain (LCEL) que recibe un párrafo de texto sin procesar (un log de error, un reporte de incidente o una descripción de arquitectura) y devuelve un objeto **validado con Pydantic** con las tecnologías involucradas, el nivel de criticidad y un resumen técnico.

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Estructura del repositorio

| Archivo | Qué contiene |
|---|---|
| [`schemas.py`](schemas.py) | Contrato de salida: modelo Pydantic `ExtraccionTecnica` y enum `NivelCriticidad` (`baja`, `media`, `alta`). |
| [`chain.py`](chain.py) | Cliente LLM (OpenAI o Anthropic), `ChatPromptTemplate`, cadena LCEL con `.with_structured_output()` y `.with_retry()`, validación del `finish_reason`, logs y la función asíncrona `process_text()`. |
| [`prueba_async.py`](prueba_async.py) | Mini-script de prueba asíncrono: procesa 3 textos en paralelo, uno de ellos ambiguo (prueba de estrés). |
| [`tests/`](tests) | 35 tests con pytest (esquema, prompt, reintentos, truncado, logs y concurrencia) que corren sin API key. |
| `requirements.txt`, `.env.example` | Dependencias y variables de configuración. |

## Cómo funciona

```
              ┌──────────────────────────── .with_retry(stop_after_attempt=3) ────────────────────────────┐
 {"texto"} ─► │ ChatPromptTemplate ─► llm.with_structured_output(ExtraccionTecnica) ─► validar_salida     │ ─► ExtraccionTecnica
              │  {texto}                 method="function_calling", include_raw=True     · finish_reason   │
              │  {instrucciones_formato}                                                  · parsing_error   │
              │                                                                           · parsed is None  │
              └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Esquema Pydantic** ([`schemas.py`](schemas.py)): además de los tipos, el esquema fija restricciones que el LLM debe cumplir:
   - `tecnologias`: entre 1 y 20 elementos; se quitan espacios, cadenas vacías y duplicados (sin distinguir mayúsculas). Si después de limpiar la lista queda vacía, el objeto se rechaza.
   - `nivel_de_criticidad`: solo `baja`, `media` o `alta`.
   - `resumen_tecnico`: entre 15 y 400 caracteres.
   - `extra="forbid"`: se rechaza cualquier campo no declarado.
   - Las descripciones de cada campo forman parte del esquema que recibe el modelo, así que también le sirven de instrucciones.

2. **Prompt template** (`build_prompt()` en [`chain.py`](chain.py)): un `ChatPromptTemplate` con dos variables, `{texto}` y `{instrucciones_formato}`. Las instrucciones de formato se generan desde el propio esquema (`generar_instrucciones_formato()`) y se fijan con `.partial()`, de modo que el prompt y el contrato no se desincronizan. No hay f-strings dentro de la cadena: LangChain gestiona las variables. El texto del usuario va entre etiquetas `<texto>` y el prompt de sistema indica tratarlo como dato, no como instrucciones (mitiga la inyección de prompts).

3. **Cadena LCEL** (`build_chain()`):

   ```python
   cadena = (
       prompt
       | llm.with_structured_output(ExtraccionTecnica, method="function_calling", include_raw=True)
       | RunnableLambda(validar_salida)
   )
   ```

   - `include_raw=True` conserva el `AIMessage` original junto al objeto parseado. Así se puede revisar el **`finish_reason`** antes de aceptar el objeto: si el modelo cortó la respuesta por falta de tokens (`finish_reason="length"` en OpenAI o `stop_reason="max_tokens"` en Anthropic), el objeto se descarta aunque parezca válido.
   - `method="function_calling"` funciona igual con OpenAI y con Anthropic, y evita las limitaciones del modo `strict` de OpenAI con `minItems` y `minLength`. Esas restricciones las valida Pydantic en el cliente.

4. **Lógica de resiliencia**: `validar_salida` lanza `RespuestaInvalidaError` cuando:
   - la respuesta está truncada (`finish_reason`),
   - el JSON viene mal formado o el modelo no llamó a la herramienta (`parsed is None`),
   - el objeto no cumple el esquema (campo faltante, lista vacía, enum inválido, etc.).

   La cadena completa está envuelta en `.with_retry(retry_if_exception_type=(RespuestaInvalidaError,), stop_after_attempt=3)` con espera exponencial y jitter. Solo se reintentan los errores de formato. Los errores de red y de *rate limit* los reintenta el SDK del proveedor (`max_retries=2`). Si el modelo **se niega** a responder (`refusal`), no se reintenta, porque repetir la misma pregunta no cambiaría el resultado.

5. **Función asíncrona** `process_text(text: str)`: valida la entrada (no vacía y de 8.000 caracteres como máximo) y ejecuta la cadena con `.ainvoke()`. Devuelve un `ExtraccionTecnica`. Si se agotan los reintentos, lanza `ExtraccionFallidaError`.

6. **Logs**: `.with_retry()` no emite eventos propios, así que un *callback handler* (`RegistroDeIntentos`) cuenta las llamadas al LLM y registra cada intento, reintento, `finish_reason` y tokens usados. `validar_salida` registra por qué se rechazó cada respuesta. Cada línea lleva la etiqueta del texto para poder seguir varias ejecuciones en paralelo.

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # y completar la API key
```

Variables de `.env`:

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | (sin default) | Clave del proveedor elegido |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Modelo de Anthropic |
| `LLM_TEMPERATURE` | `0` | Temperatura (0 = salida determinista, ideal para extracción) |
| `LLM_MAX_TOKENS` | `1024` | Límite de tokens de salida |
| `LLM_TIMEOUT` | `30` | Timeout por llamada, en segundos |

## Uso

### Mini-script de prueba asíncrono

```bash
python prueba_async.py                  # con el LLM real configurado en .env
python prueba_async.py --max-tokens 30  # prueba de estrés: fuerza respuestas truncadas
python prueba_async.py --offline        # sin API key: modelo falso que falla a propósito
```

El script procesa tres textos en paralelo con `asyncio.gather`:

- `arquitectura`: descripción de una API FastAPI + Nginx + Redis + PostgreSQL con problemas de conexiones.
- `log_error`: log de Celery con `psycopg2.OperationalError`, un pod en `CrashLoopBackOff` y una cola de RabbitMQ acumulada.
- `ambiguo` (prueba de estrés): *"Ayer volvió a pasar lo de siempre: algo se puso lento, la gente se quejó y después se arregló solo..."*. No menciona ninguna tecnología, así que permite observar si el modelo se recupera o si el validador termina lanzando la excepción.

### Desde código

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text(
    "Nuestro servicio en Django usa Celery con Redis como broker y se cae al procesar lotes grandes."
))
print(resultado.model_dump_json(indent=2))
```

### Tests

```bash
python -m pytest
```

Los tests usan `tests/fake_llm.py`, un modelo de chat falso con respuestas guionadas que implementa *tool calling*. Con él se reproducen, sin gastar tokens, los casos de JSON mal formado, objeto incompleto, lista vacía, enum inválido, respuesta sin *tool call*, truncado (OpenAI y Anthropic), negativa del modelo y reintentos agotados.

## Ejemplo de ejecución

### Con OpenAI (`gpt-4o-mini`)

Salida real de `python prueba_async.py`:

```text
INFO     pipeline | Usando OpenAI (gpt-4o-mini)
INFO     pipeline | [ambiguo] LLM respondió (finish_reason=stop, tokens_salida=49)
WARNING  pipeline | [ambiguo] Validación Pydantic fallida: 1 validation error for ExtraccionTecnica | tecnologias | List should have at least 1 item after validation, not 0 [type=too_short, ...]
INFO     pipeline | [arquitectura] LLM respondió (finish_reason=stop, tokens_salida=94)
INFO     pipeline | [arquitectura] Validación OK: 4 tecnología(s), criticidad=media
INFO     pipeline | [arquitectura] Extracción válida en 1 intento(s)
INFO     pipeline | [log_error] LLM respondió (finish_reason=stop, tokens_salida=101)
INFO     pipeline | [log_error] Validación OK: 4 tecnología(s), criticidad=alta
INFO     pipeline | [log_error] Extracción válida en 1 intento(s)
WARNING  pipeline | [ambiguo] Reintento automático: intento 2
WARNING  pipeline | [ambiguo] Validación Pydantic fallida: ...
WARNING  pipeline | [ambiguo] Reintento automático: intento 3
WARNING  pipeline | [ambiguo] Validación Pydantic fallida: ...
ERROR    pipeline | [ambiguo] Sin objeto válido tras 3 intento(s): El objeto devuelto no cumple el esquema: ...
```

```json
### arquitectura
{
  "tecnologias": ["FastAPI", "Nginx", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "La API, construida con FastAPI y detrás de Nginx, experimenta problemas de rendimiento en horas pico debido a que el pool de conexiones de PostgreSQL se agota con más de 500 requests concurrentes, lo que causa un aumento en la latencia y timeouts en el frontend."
}

### log_error
{
  "tecnologias": ["Celery", "PostgreSQL", "RabbitMQ", "Kubernetes"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Se ha producido un error operativo en la tarea de Celery 'billing.charge_invoice' debido a que se han agotado los slots de conexión disponibles en PostgreSQL, lo que impide nuevas conexiones. Esto ha llevado a un fallo en el contenedor 'billing-worker' en Kubernetes, que está en un estado de reinicio continuo."
}

### ambiguo
FALLÓ LA VALIDACIÓN -> No se obtuvo un objeto válido tras 3 intento(s): ...

Resumen: 2/3 textos con objeto válido
```

Prueba de estrés: el texto ambiguo no menciona tecnologías, así que el modelo devuelve una lista vacía. El validador la rechaza en los 3 intentos y el pipeline lanza la excepción en lugar de devolver un objeto inválido.

Con `python prueba_async.py --max-tokens 30`, el modelo se queda sin tokens a mitad del objeto y el pipeline lo detecta por el `finish_reason`:

```text
INFO     pipeline | [arquitectura] LLM respondió (finish_reason=length, tokens_salida=30)
WARNING  pipeline | [arquitectura] Respuesta truncada por límite de tokens (finish_reason=length); se descarta.
WARNING  pipeline | [arquitectura] Reintento automático: intento 2
...
ERROR    pipeline | [arquitectura] Sin objeto válido tras 3 intento(s): La respuesta del LLM quedó incompleta (finish_reason=length).
```

### Sin API key (`--offline`)

Extracto de la salida de `python prueba_async.py --offline`. El modelo falso falla a propósito en el primer intento de cada texto:

```text
INFO     pipeline | [arquitectura] Intento 1: llamando al LLM
INFO     pipeline | [log_error] Intento 1: llamando al LLM
INFO     pipeline | [ambiguo] Intento 1: llamando al LLM
WARNING  pipeline | [arquitectura] El LLM no devolvió un objeto estructurado (JSON mal formado o ausente).
WARNING  pipeline | [log_error] Respuesta truncada por límite de tokens (finish_reason=length); se descarta.
WARNING  pipeline | [arquitectura] Reintento automático: intento 2
WARNING  pipeline | [ambiguo] Validación Pydantic fallida: 1 validation error for ExtraccionTecnica | tecnologias | List should have at least 1 item after validation, not 0 [type=too_short, ...]
WARNING  pipeline | [log_error] Reintento automático: intento 2
INFO     pipeline | [arquitectura] Validación OK: 4 tecnología(s), criticidad=alta
INFO     pipeline | [arquitectura] Extracción válida en 2 intento(s)
INFO     pipeline | [log_error] Validación OK: 4 tecnología(s), criticidad=alta
INFO     pipeline | [log_error] Extracción válida en 2 intento(s)
WARNING  pipeline | [ambiguo] Reintento automático: intento 3
WARNING  pipeline | [ambiguo] Validación Pydantic fallida: 1 validation error for ExtraccionTecnica | tecnologias | ...
ERROR    pipeline | [ambiguo] Sin objeto válido tras 3 intento(s): El objeto devuelto no cumple el esquema: ...
```

```json
### arquitectura
{
  "tecnologias": ["FastAPI", "Nginx", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API FastAPI tras Nginx con caché en Redis y PostgreSQL; el pool de conexiones se agota en horas pico y dispara timeouts."
}

### log_error
{
  "tecnologias": ["Celery", "PostgreSQL", "Kubernetes", "RabbitMQ"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Workers de Celery agotan las conexiones de PostgreSQL, el pod entra en CrashLoopBackOff y la cola de RabbitMQ acumula mensajes."
}

### ambiguo
FALLÓ LA VALIDACIÓN -> No se obtuvo un objeto válido tras 3 intento(s): ...

Resumen: 2/3 textos con objeto válido
```

En el modo `--offline` las respuestas son guionadas, para mostrar cada camino de la lógica de resiliencia. Con un LLM real, el texto ambiguo puede terminar de dos maneras: el modelo infiere alguna tecnología y el objeto pasa la validación, o devuelve una lista vacía, el validador la rechaza y, tras 3 intentos, `process_text` lanza `ExtraccionFallidaError`. En ambos casos el pipeline nunca devuelve un objeto que no cumpla el esquema.

## Checklist de entrega

- [x] Repositorio con `schemas.py` (modelo Pydantic) y `chain.py` (cadena LCEL).
- [x] Cadena LCEL: `prompt | model.with_structured_output(Schema)` (`build_chain()`).
- [x] Reintento con `.with_retry()` ante JSON mal formado o incompleto, incluida la detección de truncado por `finish_reason`.
- [x] Función asíncrona `process_text()` con `.ainvoke()` y logs de validación y reintentos.
- [x] `ChatPromptTemplate` modular con `{texto}` e `{instrucciones_formato}`, sin f-strings en la cadena.
- [x] Mini-script de prueba asíncrono (`prueba_async.py`) con prueba de estrés (texto ambiguo y `--max-tokens`).
