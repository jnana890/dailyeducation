# vectordb/faiss_client.py

import os
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

VECTOR_BASE_PATH = "vector_store"

def create_faiss_index(docs: List[Document], embedding_model: Embeddings) -> FAISS:
    return FAISS.from_documents(docs, embedding_model,normalize_L2=True)

def save_faiss_index(index: FAISS, path: str):
    # Ensure path is treated as a folder (remove .faiss extension if mistakenly included)
    if path.endswith(".faiss"):
        path = path[:-6]
    index.save_local(folder_path=path)

def load_faiss_index(embedding_model: Embeddings, board: str, standard: str, subject: str) -> FAISS:
    folder = f"{VECTOR_BASE_PATH}/{board}_{standard}"
    index_folder = os.path.join(folder, f"{board}_{standard}_{subject}")

    if not os.path.exists(index_folder):
        raise FileNotFoundError(f"No FAISS index found at {index_folder}")

    return FAISS.load_local(
        folder_path=index_folder,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

def load_docs_metadata(board: str, standard: str, subject: str) -> List[Document]:
    folder = f"{VECTOR_BASE_PATH}/{board}_{standard}"
    pkl_path = os.path.join(folder, f"{board}_{standard}_{subject}.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No metadata file found at {pkl_path}")

    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def get_retriever(index: FAISS, k: int = 5):
    return index.as_retriever(search_kwargs={"k": k})
