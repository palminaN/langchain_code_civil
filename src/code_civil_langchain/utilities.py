import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openrouter_embeddings import CustomEmbedOpenRouter
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

root_dir = os.getenv("ROOT_DIR")


vector_store = Chroma(
    collection_name="articles_code_civil",
    embedding_function=CustomEmbedOpenRouter(api_key=api_key),
    persist_directory=root_dir + "/chroma_langchain_db",
)


class ArticlesQueries(BaseModel):
    is_rag: bool = Field(
        "Given that we have the full Code civil in database, set this variable to True if user question can be answered using RAG. If you estimate this is not the case then set all other variables to None and this one to False"
    )
    query1: str | None = Field(
        description="Rephrasing of client question to query the vector database"
    )
    query2: str | None = Field(description="Second rephrasing of client question")
    query3: str | None = Field(description="Third rephrasing of client question")
    keywords: list[str] | None = Field(
        description="List of keywords relevant to the client question to find better matches in the vector database"
    )


def query_documents(article_queries: ArticlesQueries) -> list[Document]:

    matches = []

    kw = article_queries.keywords
    kw = " ".join(kw)

    queries = [
        article_queries.query1 + kw,
        article_queries.query2 + kw,
        article_queries.query3 + kw,
    ]

    for q in queries:

        search_results = vector_store.similarity_search(q, k=2)
        matches = matches + search_results

    return matches
