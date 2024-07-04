from langchain.globals import set_debug
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

# set_debug(True)

chat = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={
        "max_tokens": 1000,
    },
    streaming=True,
)

messages = [
    SystemMessage(content="Act as an alien"),
    HumanMessage(content="What is human?"),
]

response = chat.stream(messages)
for chunk in response:
    print(chunk.content, end="", flush=True)

print("")
