from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrockConverse
from typing import List, Dict, Any
from pydantic import BaseModel
import boto3
from botocore.exceptions import NoCredentialsError
import requests


def get_models_for_chatbots(app: str, is_testing: bool) -> dict:
    url = "https://intranet.ufm.edu/asistente_procesos_api.php"
    params = {
        "getModelsForChatbots": "true",
        "app": app
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()

    data = r.json()

    if not data.get("success"):
        raise RuntimeError("Error al obtener modelos")

    model_chat = None
    model_rename = None

    for row in data["data"]:
        if row["TIPO"] == "CHAT":
            model_chat = (
                row["MODEL_ID_BEDROCK"]
                if is_testing
                else row["MODEL_INFERENCE_PROFILE"]
            )

        elif row["TIPO"] == "RENAME":
            model_rename = (
                row["MODEL_ID_BEDROCK"]
                if is_testing
                else row["MODEL_INFERENCE_PROFILE"]
            )

    if not model_chat or not model_rename:
        raise RuntimeError("Faltan modelos CHAT o RENAME")

    return {
        "CHAT": model_chat,
        "RENAME": model_rename
    }


IS_TESTING = False

models = get_models_for_chatbots(app="CHH", is_testing=IS_TESTING)

model_id_chat = models["CHAT"]
model_id_rename = models["RENAME"]

session = boto3.Session(profile_name="testing" if IS_TESTING else None)

bedrock_runtime = session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

model = ChatBedrockConverse(
    client=bedrock_runtime,
    model_id=model_id_chat,
    max_tokens=4096,
    temperature=0.0,
    additional_model_request_fields={
        "top_k": 250
    },
    provider="anthropic",
    disable_streaming=False,
)

modelNames = ChatBedrockConverse(
    client=bedrock_runtime,
    model_id=model_id_rename,
    max_tokens=256,
    temperature=0.0,
    additional_model_request_fields={
        "top_k": 250
    },
    provider="anthropic"
)


def history_to_text(history: Any) -> str:
    if not history:
        return ""

    lines = []

    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "user")
            content = getattr(msg, "content", "")

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def normalize_history_for_converse(history: Any) -> list:
    if not history:
        return []

    normalized = []

    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "user")
            content = getattr(msg, "content", "")

        if role not in ["user", "assistant"]:
            role = "user"

        normalized.append((role, content))

    return normalized


def docs_to_context(docs) -> str:
    bloques = []

    for i, doc in enumerate(docs, start=1):
        bloques.append(f"[Fragmento {i}]")
        bloques.append(doc.page_content)

        if doc.metadata:
            bloques.append(f"Metadata: {doc.metadata}")

        bloques.append("")

    return "\n".join(bloques)


def get_text_from_response(response) -> str:
    text_attr = getattr(response, "text", None)

    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr.strip()

    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        partes = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    partes.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    partes.append(text)

        if partes:
            return "\n".join(partes).strip()

    return str(response).strip()


def get_text_from_chunk(chunk) -> str:
    text_attr = getattr(chunk, "text", None)

    if isinstance(text_attr, str):
        return text_attr

    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        partes = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    partes.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    partes.append(text)

        return "".join(partes)

    return ""


SYSTEM_PROMPT_HAYEK = """
# Prompt del Sistema: Chatbot Especializado en Friedrich A. Hayek

## **Identidad del Asistente**
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Friedrich A. Hayek y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos complejos desarrollados por Hayek mediante interacciones estructuradas y personalizadas. Destacas por tu capacidad de compilar y sintetizar información precisa sobre las teorías de Hayek, respondiendo en español e inglés.
"""

# Base de conocimiento en Bedrock
BASE_CONOCIMIENTOS_HAYEK = "HME7HA8YXX"

retriever_hayek = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=BASE_CONOCIMIENTOS_HAYEK,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 25}},
)

# RERANKING, us-west-2
retriever_hayek_RERANK = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=BASE_CONOCIMIENTOS_HAYEK,
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 25,
            "rerankingConfiguration": {
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
                    },
                    "numberOfRerankedResults": 10
                },
                "type": "BEDROCK_RERANKING_MODEL"
            }
        }
    }
)


def run_hayek_chain(question, history):
    docs = retriever_hayek.invoke(question)
    context_text = docs_to_context(docs)
    normalized_history = normalize_history_for_converse(history)

    messages = [
        (
            "system",
            [
                {"type": "text", "text": SYSTEM_PROMPT_HAYEK},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
                {
                    "type": "text",
                    "text": f"## Información relevante recuperada para esta pregunta:\n{context_text}",
                },
            ],
        )
    ]

    messages.extend(normalized_history)
    messages.append(("human", question))

    last_usage_metadata = None

    for chunk in model.stream(messages):
        chunk_text = get_text_from_chunk(chunk)
        chunk_usage = getattr(chunk, "usage_metadata", None)

        if chunk_usage:
            last_usage_metadata = chunk_usage

        if chunk_text:
            yield {
                "response": chunk_text,
                "context": docs,
                "usage_metadata": None,
            }

    yield {
        "response": "",
        "context": docs,
        "usage_metadata": last_usage_metadata,
    }


##################################################################################
### HAZLITT, PROMPT Y CHAIN
SYSTEM_PROMPT_HAZLITT = """
# Prompt del Sistema: Chatbot Especializado en Henry Hazlitt

## **Identidad del Asistente**
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Henry Hazlitt y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos complejos desarrollados por Henry Hazlitt mediante interacciones estructuradas y personalizadas. Destacas por tu capacidad de compilar y sintetizar información precisa sobre las teorías de Henry Hazlitt, respondiendo en español e inglés.
"""

BASE_CONOCIMIENTOS_HAZLITT = "7MFCUWJSJJ"

retriever_hazlitt = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=BASE_CONOCIMIENTOS_HAZLITT,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 25}},
)


def run_hazlitt_chain(question, historial):
    docs = retriever_hazlitt.invoke(question)
    context_text = docs_to_context(docs)
    normalized_history = normalize_history_for_converse(historial)

    messages = [
        (
            "system",
            [
                {"type": "text", "text": SYSTEM_PROMPT_HAZLITT},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
                {
                    "type": "text",
                    "text": f"## Información relevante recuperada para esta pregunta:\n{context_text}",
                },
            ],
        )
    ]

    messages.extend(normalized_history)
    messages.append(("human", question))

    last_usage_metadata = None

    for chunk in model.stream(messages):
        chunk_text = get_text_from_chunk(chunk)
        chunk_usage = getattr(chunk, "usage_metadata", None)

        if chunk_usage:
            last_usage_metadata = chunk_usage

        if chunk_text:
            yield {
                "response": chunk_text,
                "context": docs,
                "usage_metadata": None,
            }

    yield {
        "response": "",
        "context": docs,
        "usage_metadata": last_usage_metadata,
    }


#############################################################
# MISES, PROMPT Y CHAIN

SYSTEM_PROMPT_MISES = """
# Prompt del Sistema: Chatbot Especializado en Ludwig von Mises

## **Identidad del Asistente**
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Ludwig von Mises y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos complejos desarrollados por Ludwig von Mises mediante interacciones estructuradas y personalizadas. Destacas por tu capacidad de compilar y sintetizar información precisa sobre las teorías de Ludwig von Mises, respondiendo en español e inglés.

Este asistente también cumple el rol de tutor complementario para cursos de la Universidad Francisco Marroquín (UFM), donde todos los estudiantes deben cursar materias como Filosofía de Mises, Ética de la libertad, Economía Austriaca 1 y 2, entre otras relacionadas.

## Contexto Pedagógico y Estilo Empático
"""

BASE_CONOCIMIENTOS_MISES = "4L0WE8NOOH"

retriever_mises = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=BASE_CONOCIMIENTOS_MISES,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 25}},
)


def run_mises_chain(question, historial):
    docs = retriever_mises.invoke(question)
    context_text = docs_to_context(docs)
    normalized_history = normalize_history_for_converse(historial)

    messages = [
        (
            "system",
            [
                {"type": "text", "text": SYSTEM_PROMPT_MISES},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
                {
                    "type": "text",
                    "text": f"## Información relevante recuperada para esta pregunta:\n{context_text}",
                },
            ],
        )
    ]

    messages.extend(normalized_history)
    messages.append(("human", question))

    last_usage_metadata = None

    for chunk in model.stream(messages):
        chunk_text = get_text_from_chunk(chunk)
        chunk_usage = getattr(chunk, "usage_metadata", None)

        if chunk_usage:
            last_usage_metadata = chunk_usage

        if chunk_text:
            yield {
                "response": chunk_text,
                "context": docs,
                "usage_metadata": None,
            }

    yield {
        "response": "",
        "context": docs,
        "usage_metadata": last_usage_metadata,
    }


##################################################################################
# TODOS LOS AUTORES === Prompt y cadena para Todos los Autores ===
SYSTEM_PROMPT_GENERAL = """
# Prompt del Sistema: Chatbot Especializado en Hayek, Hazlitt, Mises y Manuel F. Ayau (Muso)
"""

BASE_CONOCIMIENTOS_GENERAL = "WGUUTHDVPH"

retriever_general = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=BASE_CONOCIMIENTOS_GENERAL,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 25}},
)


def run_general_chain(question, historial):
    docs = retriever_general.invoke(question)
    context_text = docs_to_context(docs)
    normalized_history = normalize_history_for_converse(historial)

    messages = [
        (
            "system",
            [
                {"type": "text", "text": SYSTEM_PROMPT_GENERAL},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
                {
                    "type": "text",
                    "text": f"## Información relevante recuperada para esta pregunta:\n{context_text}",
                },
            ],
        )
    ]

    messages.extend(normalized_history)
    messages.append(("human", question))

    last_usage_metadata = None

    for chunk in model.stream(messages):
        chunk_text = get_text_from_chunk(chunk)
        chunk_usage = getattr(chunk, "usage_metadata", None)

        if chunk_usage:
            last_usage_metadata = chunk_usage

        if chunk_text:
            yield {
                "response": chunk_text,
                "context": docs,
                "usage_metadata": None,
            }

    yield {
        "response": "",
        "context": docs,
        "usage_metadata": last_usage_metadata,
    }


## reformulador_interno

REFORMULATE_WITH_HISTORY_PROMPT_MUSO = """
Actúa como un reformulador de preguntas para un asistente virtual que responde exclusivamente en la voz de Manuel F. Ayau (Muso), economista guatemalteco defensor del liberalismo clásico. 

Tu tarea es transformar la última pregunta del usuario en una versión clara, autosuficiente y redactada como una **instrucción explícita** para que el asistente responda en **primera persona** (como Muso), usando un estilo **narrativo, directo, lógico y con ejemplos cotidianos**.

Toma en cuenta el historial del chat para entender el contexto.

Guías para reformular:
- Si la pregunta es impersonal o genérica (ej. “¿Qué principios guían su pensamiento económico?”), conviértela en una instrucción como:  
  “Explica desde la perspectiva de Muso, en primera persona, qué principios éticos guiaban su pensamiento económico.”
- Si la pregunta es biográfica (ej. “¿Cuándo fundó el CEES?”), conviértela en:  
  “Relata en primera persona por qué, cuándo y cómo fundaste el CEES, incluyendo la motivación detrás.”
- Si el usuario pide opinión (ej. “¿Qué piensa sobre la redistribución?”), reformula como:  
  “Explica en primera persona por qué Muso está en contra de la redistribución forzada, con argumentos éticos y económicos.”
- Si el input ya está formulado en primera persona o dirigido correctamente, respétalo tal como está.
- Si hay ambigüedad o informalidad, aclara el foco y fuerza el uso de primera persona narrativa (por ejemplo: “¿Cómo veía Muso la educación?” → “Cuenta cómo entendías el rol de la educación en una sociedad libre, desde tu experiencia personal.”)

Reglas adicionales:
- Siempre asegúrate de que la nueva pregunta **implique que el modelo debe hablar como Muso**, desde su punto de vista.
- Nunca uses tercera persona en la reformulación. No digas “qué pensaba Muso…”, sino “explica en primera persona…”.
- No agregues explicaciones ni comentarios, responde solo con la versión reformulada.
"""


def reformulate_question_muso(question, history):
    history_text = history_to_text(history)

    messages = [
        (
            "human",
            [
                {"type": "text", "text": REFORMULATE_WITH_HISTORY_PROMPT_MUSO},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
                {
                    "type": "text",
                    "text": (
                        f"Historial del chat:\n{history_text}\n\n"
                        f"Última pregunta del usuario:\n{question}\n\n"
                        "Pregunta reformulada:"
                    ),
                },
            ],
        )
    ]

    response = model.invoke(messages)
    return get_text_from_response(response)


###############################################################################
# PARA MUSO

SYSTEM_PROMPT_MUSO = """
# Prompt del Sistema: Chatbot Especializado en Manuel F. Ayau (Muso).

## **Identidad del Asistente**
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Manuel F. Ayau apodado Muso y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos complejos desarrollados por Manuel F. Ayau (Muso) mediante interacciones estructuradas y personalizadas. Destacas por tu capacidad de compilar y sintetizar información precisa sobre las teorías de Manuel F. Ayau (Muso), respondiendo en español e inglés.

---
"""

BASE_CONOCIMIENTOS_MUSO = "HE8WRDDBFH"

retriever_muso = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=BASE_CONOCIMIENTOS_MUSO,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 25}},
)


def run_muso_chain(question, historial):
    docs = retriever_muso.invoke(question)
    context_text = docs_to_context(docs)
    normalized_history = normalize_history_for_converse(historial)

    messages = [
        (
            "system",
            [
                {"type": "text", "text": SYSTEM_PROMPT_MUSO},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
                {
                    "type": "text",
                    "text": f"## Información relevante recuperada para esta pregunta:\n{context_text}",
                },
            ],
        )
    ]

    messages.extend(normalized_history)
    messages.append(("human", question))

    last_usage_metadata = None

    for chunk in model.stream(messages):
        chunk_text = get_text_from_chunk(chunk)
        chunk_usage = getattr(chunk, "usage_metadata", None)

        if chunk_usage:
            last_usage_metadata = chunk_usage

        if chunk_text:
            yield {
                "response": chunk_text,
                "context": docs,
                "usage_metadata": None,
            }

    yield {
        "response": "",
        "context": docs,
        "usage_metadata": last_usage_metadata,
    }


####################################################

# --------------------------
# Modelo para citar documentos recuperados

class Citation(BaseModel):
    page_content: str
    metadata: Dict


def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.page_content, metadata=doc.metadata) for doc in response]


# --------------------------
# Crear URL de descarga temporal desde S3

def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    """Genera una URL firmada para descargar un archivo de S3"""
    s3_client = boto3.client("s3")
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration
        )
    except NoCredentialsError:
        return ""
    return response


def parse_s3_uri(uri: str) -> tuple:
    """Convierte un URI s3://bucket/key en (bucket, key)"""
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key


GENERATE_NAME_PROMPT_FIJO = """
Eres un asistente especializado en generar títulos breves para conversaciones académicas.
Genera únicamente un título breve (máximo 50 caracteres, en español) adecuado para nombrar una conversación.
El título debe ser educativo, respetuoso y apropiado para un entorno universitario.
Evita completamente lenguaje ofensivo, burlas, juicios de valor negativos, insinuaciones violentas o términos discriminatorios hacia personas, instituciones o autores.
No incluyas insultos, groserías, sarcasmo ni referencias provocadoras.
En su lugar, reformula de manera informativa, neutral o académica.
Entrega solo el título, sin comillas ni explicaciones.
"""


def generate_name(prompt, author):
    try:
        author_names = {
            "hayek": "Friedrich A. Hayek",
            "hazlitt": "Henry Hazlitt",
            "mises": "Ludwig von Mises",
            "muso": "Manuel F. Ayau (Muso)",
            "general": "Hayek, Hazlitt, Mises, Muso"
        }
        autor_legible = author_names.get(author.lower(), "el pensamiento liberal clásico")

        messages = [
            (
                "human",
                [
                    {"type": "text", "text": GENERATE_NAME_PROMPT_FIJO},
                    {"cachePoint": {"type": "default", "ttl": "1h"}},
                    {
                        "type": "text",
                        "text": (
                            f"Autor o enfoque principal: {autor_legible}\n"
                            f"Texto base: {prompt}\n\n"
                            "Título:"
                        ),
                    },
                ],
            )
        ]

        response = modelNames.invoke(messages)
        return get_text_from_response(response)

    except Exception as e:
        return f"Error con la respuesta: {e}"