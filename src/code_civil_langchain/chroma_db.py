import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openrouter_embeddings import CustomEmbedOpenRouter

load_dotenv()

root_dir = os.getenv("ROOT_DIR")
api_key = os.getenv("OPENROUTER_API_KEY")


txt = ""
with open(
    "./data/processed_documents/LEGITEXT000006070721.md", "r", encoding="utf-8"
) as d:
    txt = d.read()


chunks = re.split(r"## Article", txt)


splitted = (c.split("\n", 1) for c in chunks)

dnf = [(a[0].strip(), a[1]) for a in splitted]


vector_store = Chroma(
    collection_name="articles_code_civil",
    embedding_function=CustomEmbedOpenRouter(api_key=api_key),
    persist_directory=root_dir + "/chroma_langchain_db",
)

docs = [
    Document(page_content=c[1], metadata={"source": f"Article {c[0]} du code civil"})
    for c in dnf
]

try:
    vector_store.add_documents(documents=docs)
    print("Insertion done!")
except Exception as e:  # noqa: BLE001
    print(f"Error while insertion : {e}")
