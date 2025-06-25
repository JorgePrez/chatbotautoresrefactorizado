import boto3

regiones = ['us-east-1', 'us-east-2']
tag_a_borrar = 'Chatbot'  # Sensible a mayúsculas

for region in regiones:
    print(f"\n📍 Revisando región: {region}")
    client = boto3.client('resourcegroupstaggingapi', region_name=region)

    paginator = client.get_paginator('get_resources')
    page_iterator = paginator.paginate()

    for page in page_iterator:
        for resource in page['ResourceTagMappingList']:
            arn = resource['ResourceARN']
            tags = {tag['Key']: tag['Value'] for tag in resource.get('Tags', [])}

            if tag_a_borrar in tags:
                print(f"🧹 Eliminando tag '{tag_a_borrar}' de recurso:\n🔗 {arn}")
                client.untag_resources(
                    ResourceARNList=[arn],
                    TagKeys=[tag_a_borrar]
                )
