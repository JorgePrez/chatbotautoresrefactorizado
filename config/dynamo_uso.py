import boto3
import csv
import json
import re
from datetime import datetime, timezone
from boto3.dynamodb.types import TypeDeserializer

TABLE_NAME = "CHHSessionTablePruebas"
OUTPUT_FILE = "chh_export_anonimizado_ordenado.csv"

deserializer = TypeDeserializer()


def deserialize_item(item):
    """Convierte un item de DynamoDB (AttributeValue) a dict normal."""
    return {k: deserializer.deserialize(v) for k, v in item.items()}


def extract_author(pk_value):
    """
    Extrae el autor desde un PK tipo:
    USER#correo@ufm.edu#AUTHOR#mises
    """
    if not pk_value:
        return ""

    match = re.search(r"#AUTHOR#([^#]+)$", str(pk_value))
    if match:
        return match.group(1)

    return ""


def parse_created_at(value):
    """
    Convierte CreatedAt a datetime para ordenar.
    Si falla, devuelve datetime mínimo para mandar el registro al final.
    """
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value).strip()

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return datetime.min.replace(tzinfo=timezone.utc)


def clean_text(value):
    """
    Elimina saltos de línea y colapsa espacios múltiples.
    """
    if isinstance(value, str):
        value = value.replace("\r", " ")
        value = value.replace("\n", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    return value


def normalize_value(value):
    """
    Convierte valores complejos a texto apto para CSV y limpia texto.
    """
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)

    return clean_text(value)


def sanitize_item(item):
    """
    - elimina DeletedAt e IsDeleted
    - reemplaza PK por Author
    - limpia saltos de línea en todos los campos
    """
    item = dict(item)

    pk_value = item.get("PK", "")
    author = extract_author(pk_value)

    item.pop("DeletedAt", None)
    item.pop("IsDeleted", None)
    item.pop("PK", None)

    cleaned = {"Author": author}

    for key, value in item.items():
        cleaned[key] = normalize_value(value)

    return cleaned


def scan_table(table):
    """
    Lee toda la tabla con paginación.
    """
    items = []
    scan_kwargs = {}
    page = 1

    while True:
        response = table.scan(**scan_kwargs)
        page_items = response.get("Items", [])
        items.extend(page_items)

        print(f"Página {page}: {len(page_items)} items (acumulados: {len(items)})")
        page += 1

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

        scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

    return items


def export_to_csv(items, output_file):
    """
    Exporta a CSV asegurando:
    - Author primero
    - el resto de columnas después
    - comillas en todos los campos
    """
    if not items:
        print("No hay datos para exportar.")
        return

    fieldnames = ["Author"]
    seen = {"Author"}

    for item in items:
        for key in item.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(items)

    print(f"CSV generado correctamente: {output_file}")


def main():
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(TABLE_NAME)

    print(f"Leyendo tabla: {TABLE_NAME}")
    raw_items = scan_table(table)
    print(f"Items encontrados: {len(raw_items)}")

    processed_items = [sanitize_item(item) for item in raw_items]

    processed_items.sort(
        key=lambda x: parse_created_at(x.get("CreatedAt")),
        reverse=True
    )

    export_to_csv(processed_items, OUTPUT_FILE)


if __name__ == "__main__":
    main()