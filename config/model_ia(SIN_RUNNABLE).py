

#import streamlit as st
import boto3

from langchain_aws import ChatBedrock

from langchain.schema import HumanMessage, AIMessage, SystemMessage

bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )



model_kwargs = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}


#inference_profile = "us.meta.llama3-2-3b-instruct-v1:0"

inference_profile3_5claudehaiku="us.anthropic.claude-3-5-haiku-20241022-v1:0"
inference_profile3claudehaiku="us.anthropic.claude-3-haiku-20240307-v1:0"
inference_profile3_7Sonnet="us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# us.meta.llama3-2-11b-instruct-v1:0

# Generador de respuesta ChatBedrock:
model = ChatBedrock(
    client=bedrock_runtime,
    model_id=inference_profile3_7Sonnet,
    model_kwargs=model_kwargs,
)


# Prompt del sistema
SYSTEM_PROMPT = """
Eres un experto en Hayek. Responde de forma clara y concisa sobre temas relacionados con su pensamiento económico y filosófico.
"""





def generate_response(messages):
    try:
        formatted_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_messages.append(AIMessage(content=content))
            elif role == "system":
                formatted_messages.append(SystemMessage(content=content))

        #response = model.invoke(formatted_messages)
        #return response.content
        return model.stream(formatted_messages)


    except Exception as e:
        return f"Error con la respuesta: {e}"
    

    


def generate_name(prompt):
    try:
        input_text = f"Genera un nombre en base a este texto: {prompt} no superior a 50 caracteres."
        response = model.invoke(input_text)
        return response.content
    except Exception as e:
        return f"Error con la respuesta: {e}"
