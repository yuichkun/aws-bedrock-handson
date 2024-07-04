import base64
import json
import boto3

bedrock_runtime = boto3.client("bedrock-runtime")

with open("orange.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
body = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": "You must answer as a character from a classic RPG game.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "What's the possibility that this is an apple?",
                    },
                ],
            },
        ],
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
