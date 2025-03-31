import boto3
import json
from botocore.exceptions import ClientError
from datetime import datetime
import config.model_ia as model  # para usar model.generate_name

# Inicializar recurso de DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("CHHSessionTablePruebas")

# Simulación simple (puedes reemplazar con acceso a tabla "users" si la usas)
def getUser(user_id):
    return user_id

def save(chat_id, user_id, name, chat):
    item = {
        "PK": f"USER#{user_id}",
        "SK": f"CHAT#{chat_id}",
        "Name": name,
        "Chat": chat,
        "CreatedAt": datetime.utcnow().isoformat()
    }
    table.put_item(Item=item)

def edit(chat_id, chat, user_id):
    table.update_item(
        Key={"PK": f"USER#{user_id}", "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET Chat = :chat",
        ExpressionAttributeValues={":chat": chat}
    )

def getChats(user_id):
    try:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("PK").eq(f"USER#{user_id}"),
            ScanIndexForward=False  # Orden descendente
        )
        data = response.get("Items", [])
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

def delete(chat_id, user_id):
    table.delete_item(
        Key={"PK": f"USER#{user_id}", "SK": f"CHAT#{chat_id}"}
    )

def editName(chat_id, prompt, user_id):
    name = model.generate_name(prompt)
    
    table.update_item(
        Key={"PK": f"USER#{user_id}", "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET #n = :name",
        ExpressionAttributeNames={"#n": "Name"},
        ExpressionAttributeValues={":name": name}
    )

def editNameManual(chat_id, new_name, user_id):
    table.update_item(
        Key={"PK": f"USER#{user_id}", "SK": f"CHAT#{chat_id}"},
        UpdateExpression="SET #n = :name",
        ExpressionAttributeNames={"#n": "Name"},
        ExpressionAttributeValues={":name": new_name}
    )

def getNameChat(chat_id, user_id):
    try:
        response = table.get_item(
            Key={"PK": f"USER#{user_id}", "SK": f"CHAT#{chat_id}"}
        )
        return response["Item"]["Name"]
    except KeyError:
        return None