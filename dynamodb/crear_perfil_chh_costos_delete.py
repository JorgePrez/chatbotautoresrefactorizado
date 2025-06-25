

import boto3

client = boto3.client("bedrock", region_name="us-east-1")

# ID del perfil, no el ARN completo
profile_id = "xao45s96u39n"

client.delete_inference_profile(
    inferenceProfileIdentifier=profile_id
)

print("🗑️ Perfil eliminado correctamente.")
