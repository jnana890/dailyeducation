# app/query.py

from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

from embeddings.hf_embeddings import get_embedding_model
from vectordb.faiss_client import load_faiss_index
from app.web_agent import get_fallback_answer

import wikipedia

# === Global setup ===
embedding_model = get_embedding_model()
print(f"[EMBEDDING INIT] Using model with normalize_embeddings=True")  # DEBUG

base_llm = Ollama(
    model="llama3",
    system="You are a helpful educational assistant. Respond clearly, accurately, and conversationally."
)

# Memory store
memory_store = {}

def is_unhelpful(answer: str) -> bool:
    if not answer:
        return True

    lower = answer.lower()
    return (
        "i don't know" in lower or
        "as an ai" in lower or
        "no context" in lower or
        "i don't have access" in lower or
        len(answer.strip()) < 20
    )

def get_memory_chain(chat_id: str, filters: dict = None):
    board = (filters.get("board") or "cbse").lower()
    standard = (filters.get("class") or "class10").lower()
    subject = (filters.get("subject") or "science").lower()

    print(f"[FILTERS] board={board}, class={standard}, subject={subject}")  # DEBUG

    key = f"{chat_id}_{board}_{standard}_{subject}"

    if key not in memory_store:
        print(f"[MEMORY INIT] Initializing chain for: {key}")  # DEBUG
        try:
            retriever = load_faiss_index(
                embedding_model,
                board=board,
                standard=standard,
                subject=subject
            ).as_retriever(
                search_kwargs={"k": 5}  # ❌ Removed score_threshold
            )
        except Exception as e:
            print(f"[ERROR] Failed to load FAISS index: {e}")
            raise

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=base_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

        memory_store[key] = chain

    return memory_store[key]

def chat_with_fallback(message: str, chat_id: str = "default", filters: dict = None) -> str:
    if len(message.strip().split()) <= 2:
        print("[Shortcut] Detected short input. Using LLM directly.")
        return base_llm.invoke(f"You are a helpful assistant.\nUser: {message}\nAssistant:")

    try:
        chain = get_memory_chain(chat_id, filters)
        result = chain.invoke({"question": message})
        answer = result.get("answer", "")
        sources = result.get("source_documents", [])
    except Exception as e:
        print(f"[ERROR] FAISS chain invoke failed: {e}")
        answer, sources = "", []
        fallback_due_to_faiss = True
    else:
        fallback_due_to_faiss = is_unhelpful(answer) or not sources

        print(f"[DEBUG] Retrieved {len(sources)} source documents")  # DEBUG

        for idx, doc in enumerate(sources):
            page = doc.metadata.get("page_number", "?")
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"  🔹 Source {idx+1}: Page={page}, Text='{snippet}'")

    if fallback_due_to_faiss:
        print(f"[FAISS FALLBACK] Triggered for: '{message}'")
        print("[Fallback 1] Trying Wikipedia...")
        try:
            search_results = wikipedia.search(message)
            page_title = None

            preferred_titles = ["Narendra Modi", "Prime Minister of India"]
            for preferred in preferred_titles:
                if preferred in search_results:
                    page_title = preferred
                    break

            if not page_title and search_results:
                page_title = search_results[0]

            if page_title:
                print(f"[Wikipedia] Using page: {page_title}")
                summary = wikipedia.summary(page_title, sentences=2)

                if not is_unhelpful(summary):
                    return f"From Wikipedia:\n{summary}"
                else:
                    print("[Wikipedia] Summary too vague, trying LLM...")
                    answer = ""
            else:
                print("[Wikipedia] No results found.")
                answer = ""

        except Exception as e:
            print(f"[Wikipedia Error] {e}")
            answer = ""

    # === Internal LLM fallback ===
    if is_unhelpful(answer):
        print("[Fallback 2] Using internal LLM...")
        try:
            answer = base_llm.invoke(message)
        except Exception as e:
            print(f"[LLM Error] {e}")
            answer = ""

    # === Final fallback: Web Agent ===
    if is_unhelpful(answer):
        print("[Fallback 3] All else failed. Using Web Agent...")
        return get_fallback_answer(message)

    return answer
