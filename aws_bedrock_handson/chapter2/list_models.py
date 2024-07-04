import boto3
import json

bedrock = boto3.client('bedrock')

result = bedrock.list_foundation_models()

print(
  json.dumps(
    result,
    indent=2,
    default=str
  )
)
