import json
import boto3

bedrock_runtime = boto3.client("bedrock-runtime")

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
body = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "what's bedrock?"}],
    }
)

response = bedrock_runtime.invoke_model(
    modelId=model_id,
    body=body,
    accept="application/json",
    contentType="application/json",
)
response_body = json.loads(response["body"].read())
answer = response_body["content"][0]["text"]
print(answer)
