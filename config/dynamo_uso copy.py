import boto3
import csv
import re
from boto3.dynamodb.types import TypeDeserializer

TABLE_NAME = "CHHSessionTablePruebas"
OUTPUT_FILE = "chh_export_anonimizado.csv"

deserializer = TypeDeserializer()

def deserialize_item(item):
    """Convierte un item de DynamoDB (formato AttributeValue) a dict Python normal."""
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

def sanitize_item(item):
    """
    - elimina DeletedAt e IsDeleted
    - reemplaza PK por author anonimizado
    """
    item = dict(item)

    pk_value = item.get("PK", "")
    author = extract_author(pk_value)

    # Quita los campos que no te interesan
    item.pop("DeletedAt", None)
    item.pop("IsDeleted", None)

    # Reemplaza PK por el autor anonimizado
    item["PK"] = author

    return item

def scan_table(table):
    """Lee toda la tabla con paginación."""
    items = []
    scan_kwargs = {}

    while True:
        response = table.scan(**scan_kwargs)
        raw_items = response.get("Items", [])
        items.extend(raw_items)

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

        scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

    return items

def export_to_csv(items, output_file):
    """
    Exporta una lista de diccionarios a CSV.
    Usa todas las columnas encontradas en todos los items.
    """
    if not items:
        print("No hay datos para exportar.")
        return

    # Junta todas las columnas presentes
    fieldnames = []
    seen = set()
    for item in items:
        for key in item.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)

    print(f"CSV generado correctamente: {output_file}")

def main():
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(TABLE_NAME)

    print(f"Leyendo tabla: {TABLE_NAME}")
    raw_items = scan_table(table)

    print(f"Items encontrados: {len(raw_items)}")

    processed_items = []
    for item in raw_items:
        clean_item = sanitize_item(item)
        processed_items.append(clean_item)

    export_to_csv(processed_items, OUTPUT_FILE)

if __name__ == "__main__":
    main()