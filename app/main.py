# app/main.py

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import json
import re

from app.query import chat_with_fallback
from app.ingestion import ingest_pdf_files

app = FastAPI(
    title="Educational Chatbot",
    description="Unified RAG + Memory + Metadata + Wikipedia + Web Agent chatbot",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    chat_id: str = "default"
    board: Optional[str] = None
    standard: Optional[str] = None
    subject: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "message": "What is photosynthesis?",
                "chat_id": "session_123",
                "board": "cbse",
                "standard": "class10",
                "subject": "english"
            }
        }

def clean_json_input(raw_data: bytes) -> dict:
    """Sanitize and parse JSON input"""
    try:
        return json.loads(raw_data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            cleaned = re.sub(r'[\x00-\x1F\x7F]', '', raw_data.decode('utf-8', errors='ignore'))
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid JSON format",
                    "message": str(e),
                    "suggestion": "Check for special characters or validate your JSON"
                }
            )

@app.post("/chat", response_model=dict)
async def chat_post(
    request: Request,
    chat_request: ChatRequest = Body(
        ...,
        example={
            "message": "What is photosynthesis?",
            "chat_id": "session_123",
            "board": "CBSE",
            "standard": "10",
            "subject": "Science"
        }
    )
):
    """
    Handle chat requests with robust JSON parsing and metadata filtering
    
    - **message**: Required question/message
    - **chat_id**: Optional session identifier (default: "default")
    - **board**: Optional education board filter (e.g. "CBSE")
    - **standard**: Optional class/grade filter (e.g. "10")
    - **subject**: Optional subject filter (e.g. "Science")
    """
    try:
        # Try to get raw body for validation
        try:
            body_bytes = await request.body()
            req_data = clean_json_input(body_bytes)
            chat_request = ChatRequest(**req_data)
        except Exception:
            # If raw parsing fails, use the already-parsed model
            pass
            
        # Process filters
        filters = {}
        if chat_request.board:
            filters["board"] = chat_request.board
        if chat_request.standard:
            filters["class"] = chat_request.standard
        if chat_request.subject:
            filters["subject"] = chat_request.subject

        answer = chat_with_fallback(chat_request.message, chat_request.chat_id, filters)
        return {
            "chat_id": chat_request.chat_id,
            "message": chat_request.message,
            "answer": answer,
            "filters_used": filters
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep the existing /ingest endpoint unchanged
@app.post("/ingest")
async def ingest_pdfs(
    files: List[UploadFile] = File(...),
    board: str = Form(...),
    standard: str = Form(...),
    subject: str = Form(...),
):
    filepaths = []
    for file in files:
        path = f"./uploaded_pdfs/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        filepaths.append(path)

    ingest_pdf_files(filepaths, board=board, standard=standard, subject=subject)
    return {
        "message": f"{len(filepaths)} file(s) ingested",
        "board": board,
        "class": standard,
        "subject": subject
    }