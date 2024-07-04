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

response = bedrock_runtime.invoke_model_with_response_stream(
    modelId=model_id,
    body=body,
)

for event in response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if (
        chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "text_delta"
    ):
        print(chunk["delta"]["text"], end="")

print()
