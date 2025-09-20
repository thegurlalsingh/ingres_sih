from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .database import get_db
from .schemas import ChatRequest, ChatResponse
from .i18n import translate_text_m2m, summarize_with_ollama
from .models import Assessment
from .embeddings import faiss_index, compute_embedding_for_text

router = APIRouter()


@router.post('/query', response_model=ChatResponse)
async def query_chat(req: ChatRequest, db: Session = Depends(get_db)):
    q_text = req.query

    # Translate to English if source language is not English
    if req.lang and req.lang != 'en':
        q_text = translate_text_m2m(q_text, src=req.lang, target='en')

    # Compute embedding
    q_vec = compute_embedding_for_text(q_text)

    # Load FAISS index
    if not faiss_index.load():
        return {"answer": "No data indexed yet.", "sources": []}

    # Search FAISS for top-k similar assessments
    idxs, dists = faiss_index.search(q_vec, top_k=5)
    sources = []
    top_texts = []

    for i in idxs:
        a = db.query(Assessment).filter(Assessment.id == faiss_index.ids[i]).first()
        if a:
            sources.append({
                "assessment_id": a.id,
                "year": a.year,
                "state": a.state,
                "district": a.district
            })
            top_texts.append(' '.join([f"{k}: {v}" for k, v in a.data.items()]))

    # Combine top texts into a single answer
    answer = '\n\n'.join(top_texts)

    # Summarize via Ollama
    answer = summarize_with_ollama(answer)

    # Translate back to original language if needed
    if req.lang and req.lang != 'en':
        answer = translate_text_m2m(answer, src='en', target=req.lang)

    return {"answer": answer, "sources": sources}
