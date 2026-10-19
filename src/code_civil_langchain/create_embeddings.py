import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from openrouter_embeddings import CustomEmbedOpenRouter

txt = ""
with open(
    "./data/processed_documents/LEGITEXT000006070721.md", "r", encoding="utf-8"
) as d:
    txt = d.read()


chunks = re.split(r"## Article", txt)


splitted = map(lambda c: c.split("\n", 1), chunks)

dnf = [(a[0].strip(), a[1]) for a in splitted]


store = Chroma(
    collection_name="articles_code_civil",
    embedding_function=CustomEmbedOpenRouter(),
    persist_directory=dir + "/chroma_langchain_db",
)


docs = [
    Document(page_content=c[1], metadata={"source": f"Article {c[0]} du code civil"})
    for c in dnf
]


store.add_documents(documents=docs)
