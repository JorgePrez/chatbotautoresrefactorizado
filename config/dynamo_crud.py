import boto3
import json
from botocore.exceptions import ClientError
from datetime import datetime
from boto3.dynamodb.conditions import Key, Attr
import config.model_iacatching as model  # para usar model.generate_name

# Inicializar recurso de DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("CHHSessionTablePruebas")


def getUser(user_id):
    return user_id


def build_pk(user_id, author):
    return f"USER#{user_id}#AUTHOR#{author}"


def save(chat_id, user_id, author, name, chat):
    item = {
        "PK": build_pk(user_id, author),
        "SK": f"CHAT#{chat_id}",
        "Name": name,
        "Chat": chat,
        "CreatedAt": datetime.utcnow().isoformat(),
    }
    table.put_item(Item=item)


def edit(chat_id, chat, user_id, author):
    table.update_item(
        Key={"PK": build_pk(user_id, author), "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET Chat = :chat",
        ExpressionAttributeValues={":chat": chat}
    )


def getChats(user_id, author, include_deleted=False):
    """
    - Por defecto (include_deleted=False) NO devuelve chats eliminados lógicamente.
    """
    try:
        params = {
            "KeyConditionExpression": Key("PK").eq(build_pk(user_id, author)),
            "ScanIndexForward": False  # orden por SK descendente
        }

        if not include_deleted:
            params["FilterExpression"] = Attr("IsDeleted").not_exists() | Attr("IsDeleted").eq(False)

        response = table.query(**params)
        data = response.get("Items", [])

        # Normalizar Chat a lista (por si está guardado como string JSON)
        for item in data:
            chat = item.get("Chat")
            if isinstance(chat, str):
                try:
                    item["Chat"] = json.loads(chat)
                except json.JSONDecodeError:
                    item["Chat"] = []
            elif not chat:
                item["Chat"] = []

        # Ordenar por fecha descendente (más reciente primero)
        data.sort(key=lambda x: x.get("CreatedAt", ""), reverse=True)

        return data
    except ClientError as e:
        print("Error en getChats:", e)
        return []


def deletewithChat(chat_id, user_id, author):
    table.update_item(
        Key={"PK": build_pk(user_id, author), "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET Chat = :empty, IsDeleted = :d, DeletedAt = :ts",
        ExpressionAttributeValues={
            ":empty": [],
            ":d": True,
            ":ts": datetime.utcnow().isoformat()
        }
    )

def delete(chat_id, user_id, author):
    table.update_item(
        Key={"PK": build_pk(user_id, author), "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET IsDeleted = :d, DeletedAt = :ts",
        ExpressionAttributeValues={
            ":d": True,
            ":ts": datetime.utcnow().isoformat()
        }
    )



def editName(chat_id, prompt, user_id, author):
    name = model.generate_name(prompt, author)

    table.update_item(
        Key={"PK": build_pk(user_id, author), "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET #n = :name",
        ExpressionAttributeNames={"#n": "Name"},
        ExpressionAttributeValues={":name": name}
    )


def editNameManual(chat_id, new_name, user_id, author):
    table.update_item(
        Key={"PK": build_pk(user_id, author), "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET #n = :name",
        ExpressionAttributeNames={"#n": "Name"},
        ExpressionAttributeValues={":name": new_name}
    )


def getNameChat(chat_id, user_id, author):
    try:
        response = table.get_item(
            Key={"PK": build_pk(user_id, author), "SK": f"CHAT#{chat_id}"}
        )
        return response["Item"]["Name"]
    except KeyError:
        return None

