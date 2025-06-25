import boto3

client = boto3.client("bedrock", region_name="us-east-1")
arn = "arn:aws:bedrock:us-east-1:552102268375:application-inference-profile/hkqiiam51emk"

tags = client.list_tags_for_resource(resourceARN=arn)
print("🧷 Tags aplicadas al perfil:")
for tag in tags["tags"]:
    print(f"- {tag['key']}: {tag['value']}")