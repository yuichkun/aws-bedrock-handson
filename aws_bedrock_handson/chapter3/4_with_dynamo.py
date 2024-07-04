import streamlit as st
from langchain.globals import set_debug
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Yogo Chat")

if "session_id" not in st.session_state:
    st.session_state.session_id = "session_id"

if "history" not in st.session_state:
    st.session_state.history = DynamoDBChatMessageHistory(
        table_name="BedrockChatSessionTable",
        session_id=st.session_state.session_id,
    )

if "chain" not in st.session_state:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a teacher"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="human_message"),
        ]
    )
    chat = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={
            "max_tokens": 1000,
        },
        streaming=True,
    )

    chain = prompt | chat
    st.session_state.chain = chain


if st.button("Clear chat"):
    st.session_state.history.clear()

for message in st.session_state.history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("what's up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            st.session_state.chain.stream(
                {
                    "messages": st.session_state.history.messages,
                    "human_message": [HumanMessage(content=prompt)],
                },
                config={
                    "configurable": {
                        "session_id": st.session_state.session_id,
                    }
                },
            )
        )

    st.session_state.history.add_user_message(prompt)
    st.session_state.history.add_ai_message(response)
