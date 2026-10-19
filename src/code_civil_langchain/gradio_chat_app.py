import os
import time

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openrouter import ChatOpenRouter
from utilities import ArticlesQueries, vector_store

# from langgraph.graph import START, END, StateGraph, MessagesState

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

root_dir = os.getenv("ROOT_DIR")

system_prompt = """ Tu es un assistant juriste specialie dans le droit civil.

Ton objectif est de repondre aux questions des clients en te basant sur le code civil.

IMPORTANT
Ne fais pas de supposition et n'invente RIEN qui n'est pas ecris dans le code civil.
toutes tes reponses doivent etres sourcees et contenir le ou les articles du code civil auxquels tu fais reference dans ta reponse.

FORMAT DE REPONSE
N'oubli pas que tu t'adresse a des personnes normales, possiblement en detresse et sans aucune connaissance dans le
domaine juridique. Tes reponses doivent donc etre simples, CONCISES et faciles a comprendre pour une personne lambda et avec un ton
chaleureux.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (MessagesPlaceholder(variable_name="history")),
        ("human", "message du client : {client_message}"),
    ]
)

model = ChatOpenRouter(
    model="liquid/lfm-2.5-2.6b:free", api_key=api_key, max_tokens=600, max_retries=3
)

structured_model = model.with_structured_output(ArticlesQueries)


chain = chat_prompt | model | StrOutputParser()


def slow_echo(message, history):

    # find out if rag is useful to answer user
    flow = structured_model.invoke(
        f"Is this message chat-oriented or RAG-oriented : {message}"
    )

    print(flow)
    print("\n------------------")

    if flow.is_rag:
        # Gather documents for the llm to use to make its reponse
        print("Querying Vector DB...")
        documents = []
        queries = [flow.query1, flow.query2, flow.query3]
        for q in queries:
            cat_kw = q + " ".join(flow.keywords)
            documents = documents + vector_store.similarity_search(cat_kw, k=3)
        content = "Use ONLY these documents to make your response :"

        for doc in documents:
            content = content + "\n\n" + str(doc)

        print(content)

        history.append({"role": "system", "content": content})

    bot_response = chain.invoke(input={"client_message": message, "history": history})
    for i in range(len(bot_response)):
        time.sleep(0.01)
        yield bot_response[: i + 1]


gr.ChatInterface(
    fn=slow_echo,
).launch()
