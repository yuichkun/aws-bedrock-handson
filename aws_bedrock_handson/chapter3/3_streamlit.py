import streamlit as st
from langchain.globals import set_debug
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Yogo Chat")

chat = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={
        "max_tokens": 1000,
    },
    streaming=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Act as a teacher"),
    ]

for message in st.session_state.messages:
    if message.type != "system":
        with st.chat_message(message.type):
            st.markdown(message.content)

if prompt := st.chat_input("what's up?"):
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(chat.stream(st.session_state.messages))

    st.session_state.messages.append(AIMessage(content=response))
