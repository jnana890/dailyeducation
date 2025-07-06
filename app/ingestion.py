# app/ingestion.py

import os
import magic
import pickle
from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embeddings.hf_embeddings import get_embedding_model
from vectordb.faiss_client import create_faiss_index, save_faiss_index

def validate_pdf_file(filepath: str) -> bool:
    try:
        file_type = magic.from_file(filepath, mime=True)
        if file_type != 'application/pdf':
            print(f"[VALIDATION] File is not a PDF: {file_type}")
            return False
        if os.path.getsize(filepath) < 1024:
            print(f"[VALIDATION] File too small, likely corrupted")
            return False
        return True
    except Exception as e:
        print(f"[VALIDATION ERROR] {e}")
        return False

def verify_loaded_text(docs: List[Document]) -> bool:
    if not docs:
        return False
    total_text_length = sum(len(doc.page_content) for doc in docs)
    if total_text_length < 100:
        print(f"[TEXT VERIFICATION] Insufficient text content: {total_text_length} chars")
        return False
    sample_text = docs[0].page_content[:100]
    if any(artifact in sample_text for artifact in ["�", "[PDF]", "%%EOF"]):
        print("[TEXT VERIFICATION] Found PDF extraction artifacts")
        return False
    return True

def load_pdf_text(filepath: str) -> List[Document]:
    if not validate_pdf_file(filepath):
        print(f"[SKIP] Invalid PDF file: {filepath}")
        return []

    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        print(f"[INFO] Trying UnstructuredPDFLoader...")
        docs = UnstructuredPDFLoader(filepath).load()
        if verify_loaded_text(docs):
            return docs
        print("[Fallback] Text verification failed for UnstructuredPDFLoader")
    except Exception as e1:
        print(f"[Fallback] UnstructuredPDFLoader failed: {e1}")

    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        print(f"[INFO] Trying PyMuPDFLoader...")
        docs = PyMuPDFLoader(filepath).load()
        if verify_loaded_text(docs):
            return docs
        print("[Fallback] Text verification failed for PyMuPDFLoader")
    except Exception as e2:
        print(f"[Fallback] PyMuPDFLoader failed: {e2}")

    try:
        from langchain_community.document_loaders import PDFPlumberLoader
        print(f"[INFO] Trying PDFPlumberLoader...")
        docs = PDFPlumberLoader(filepath).load()
        if verify_loaded_text(docs):
            return docs
        print("[Fallback] Text verification failed for PDFPlumberLoader")
    except Exception as e3:
        print(f"[ERROR] All loaders failed: {e3}")

    return []

def split_documents(docs: List[Document], chunk_size=500, chunk_overlap=50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def enrich_documents_with_metadata(
    docs: List[Document],
    filepath: str,
    board: Optional[str] = None,
    class_name: Optional[str] = None,
    subject: Optional[str] = None
) -> List[Document]:
    filename = os.path.basename(filepath)

    enriched_docs = []
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source_file": filename,
            "board": board or "Unknown",
            "class": class_name or "Unknown",
            "subject": subject or "Unknown",
            "book_title": filename,
            "chapter": f"Auto Chunk {i+1}",
            "chapter_title": "Auto Chunk",
            "page_number": i + 1,
        })
        enriched_docs.append(doc)

    return enriched_docs

def ingest_pdf_files(filepaths: List[str], board=None, standard=None, subject=None):
    all_docs = []

    # Normalize filters to lowercase for consistency
    board = board.lower() if board else "unknown"
    standard = standard.lower() if standard else "unknown"
    subject = subject.lower() if subject else "unknown"

    for filepath in filepaths:
        print(f"\n[START] Ingesting file: {filepath}")

        if not os.path.exists(filepath):
            print(f"[SKIP] File not found: {filepath}")
            continue

        raw_docs = load_pdf_text(filepath)

        if not raw_docs:
            print(f"[SKIP] Failed to load valid content from: {filepath}")
            continue

        split_docs = split_documents(raw_docs)
        enriched = enrich_documents_with_metadata(split_docs, filepath, board, standard, subject)
        all_docs.extend(enriched)

    if not all_docs:
        raise ValueError("❌ No documents were ingested. Please check your PDF files.")

    print(f"[INFO] Total processed chunks: {len(all_docs)}")

    embedding_model = get_embedding_model()
    index = create_faiss_index(all_docs, embedding_model)

    # === Save FAISS and metadata ===
    class_folder = f"vector_store/{board}_{standard}"
    os.makedirs(class_folder, exist_ok=True)

    index_folder = os.path.join(class_folder, f"{board}_{standard}_{subject}")
    save_faiss_index(index, index_folder)  # 🔧 pass folder path only

    metadata_path = os.path.join(class_folder, f"{board}_{standard}_{subject}.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(all_docs, f)

    print(f"[✅ SUCCESS] FAISS index saved for: {board}_{standard}_{subject}")
