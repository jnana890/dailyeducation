# embeddings/hf_embeddings.py

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns a HuggingFaceEmbeddings model instance using a lightweight SBERT model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Set to "cuda" if using GPU
        encode_kwargs={"normalize_embeddings": True}
    )
