import os
import uuid

import boto3
import streamlit as st

agent_id: str = os.getenv("BEDROCK_AGENT_ID")
if agent_id is None:
    raise ValueError("BEDROCK_AGENT_ID is not set")

agent_alias_id: str = os.getenv("BEDROCK_AGENT_ALIAS_ID")
if agent_alias_id is None:
    raise ValueError("BEDROCK_AGENT_ALIAS_ID is not set")
session_id: str = str(uuid.uuid1())
client = boto3.client("bedrock-agent-runtime")

st.title("Agent for Bedrock")

if prompt := st.chat_input("なんでも聞いてください"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = client.invoke_agent(
            inputText=prompt,
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            enableTrace=False,
        )

        event_stream = response["completion"]
        text = ""
        for event in event_stream:
            if "chunk" in event:
                text += event["chunk"]["bytes"].decode("utf-8")
        st.write(text)
