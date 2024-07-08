import os

import streamlit as st
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID")

if not knowledge_base_id:
    raise ValueError("KNOWLEDGE_BASE_ID environment variable is not set")

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=knowledge_base_id,
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 10,
        }
    },
)

prompt = ChatPromptTemplate.from_template(
    "Answer based on the following context: {context} / Question: {question}",
)

model = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={
        "max_tokens": 1000,
    },
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

st.title("Opusmodus Bot")
question = st.text_input("Ask me a question")
button = st.button("Ask")

if button:
    st.write(chain.invoke(question))
