"""JSON API for Probahini, for the Ananya mobile app.

The Streamlit page in streamlit_code.py answers a browser: it serves HTML and
drives the conversation over a WebSocket. A mobile app is not a browser — it
sends an HTTP request and expects JSON back — so a plain POST to the Streamlit
service returns 405 and the app cannot use it.

This changes none of the chatbot's behaviour. It is a thin wrapper over the
same Chain.get_response() that Streamlit already calls, exposed over HTTP so
the app can reach it.

Run:
    PYTHONPATH=/app/app uvicorn api:app --host 0.0.0.0 --port 8501
from the directory that holds vectordb/ — chain.py opens it by relative path.
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from chain import Chain

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("probahini-api")

app = FastAPI(title="Probahini API")

# Built once at import, not per request. Chain.__init__ opens the Groq client
# and the Chroma collection; doing that on every call would add seconds to
# each reply and reopen the vector store each time.
chain = Chain()


class ChatRequest(BaseModel):
    """What the Ananya app sends."""

    user_id: str
    query: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Answer one message.

    `user_id` is passed through as the chat id, which is what keys the
    conversation history inside chain.py, so a user's replies stay in context
    exactly as they do on the Streamlit page.
    """
    try:
        answer = chain.get_response(req.query, req.user_id)
        return {"response": answer}
    except Exception as exc:  # noqa: BLE001 - the app must get a reply, not a stack trace
        log.exception("get_response failed")
        return {"response": "", "error": str(exc)}


@app.get("/health")
def health():
    """Cheap endpoint for Render's health check, so it does not wake the model."""
    return {"status": "ok"}
