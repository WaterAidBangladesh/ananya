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
import os

from fastapi import FastAPI
from pydantic import BaseModel

from chain import Chain, user_history, N_RESULTS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("probahini-api")

app = FastAPI(title="Probahini API")

# Built once at import, not per request. Chain.__init__ opens the Groq client
# and the Chroma collection; doing that on every call would add seconds to
# each reply and reopen the vector store each time.
chain = Chain()

# How many times to ask before giving up.
#
# The model behind this — openai/gpt-oss-20b — reasons privately before it
# answers, and both share one output budget. Faced with the ~18,000 tokens
# chain.py attaches to every question (three whole PDF pages, because the
# vector store was built without chunking), it sometimes spends that budget
# thinking and returns an empty string. This is not an error: Groq accepts the
# request, runs it, and hands back nothing.
#
# Nor is it deterministic. The same question, with no history and temperature
# 0, was observed answering on one attempt and coming back empty on the next
# two. So asking again genuinely helps, and three attempts turn a roughly even
# chance into a rare failure.
#
# A cushion, not a cure. The cure is chunking the vector store so there is far
# less for the model to wade through; after that the first attempt should
# nearly always be the only one.
MAX_ATTEMPTS = int(os.getenv("PROBAHINI_MAX_ATTEMPTS", "3"))


class ChatRequest(BaseModel):
    """What the Ananya app sends."""

    user_id: str
    query: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Answer one message, asking again if the model returns nothing.

    `user_id` is passed through as the chat id, which is what keys the
    conversation history inside chain.py, so a user's replies stay in context
    exactly as they do on the Streamlit page.
    """
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # chain.get_response appends "user: ...\nchatbot: ..." to the history
        # whatever happens, including when the answer is empty. Left alone, a
        # retry would ask the same question against a transcript in which the
        # assistant has just said nothing — which grows the prompt and shows
        # the model an example of replying with silence. So the history is
        # snapshotted and put back whenever an attempt comes up empty.
        history_before = user_history.get(req.user_id, "")

        try:
            answer = chain.get_response(req.query, req.user_id)
        except Exception as exc:  # noqa: BLE001 - the app needs a reply, not a stack trace
            # A raised exception is a real failure — a dead key, a rate limit —
            # and asking again will not change it. Stop and report it, so the
            # app can say something accurate rather than silently retrying.
            log.exception("get_response raised on attempt %s", attempt)
            return {"response": "", "error": str(exc)}

        if answer and answer.strip():
            if attempt > 1:
                log.info("answered on attempt %s", attempt)
            return {"response": answer}

        log.warning(
            "empty answer on attempt %s of %s for user %s",
            attempt,
            MAX_ATTEMPTS,
            req.user_id,
        )
        last_error = "model returned an empty response"
        user_history[req.user_id] = history_before

    return {"response": "", "error": last_error}


@app.get("/health")
def health():
    """Cheap endpoint for Render's health check, so it does not wake the model."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Temporary diagnostic. Remove once the empty-answer question is settled.
#
# get_response() returns res.content and throws the rest of the reply away, so
# when content is empty there is nothing left to explain why. This repeats the
# same call — same retrieval, same prompt, same model — and reports what came
# back around the content: the finish reason, the token counts, and whether
# the model put its output somewhere other than `content`.
#
# It answers one question: is the model producing nothing, or producing
# something the code is not reading?
# ---------------------------------------------------------------------------

_DEBUG_TEMPLATE = """ Relevant information: {answer}

        Background: You are an expert in menstrual health topics, structured to provide information based on both
        high-level (prime) and specific (follow-up) questions. If the user message aligns with a general or
        overarching question, respond with the prime answer and, in rare occasions, suggest a couple of follow-up
        questions below it. If the question seeks specific details, provide the relevant follow-up answer. In cases
        where multiple relevant details exist, respond concisely with the most applicable information. You are
        empathetic and considerate, communicating in English or Bangla based on the user's language preference.
        If you detect a language preference from the user's message, respond accordingly. Engage in conversational
        interactions, and for questions, provide specific, accurate answers based on the relevant information below.
        Please don't share any of the question labels; only deliver the content of the answer.

        Note: You must *only* provide answers from the exact information provided in the "Relevant information"
        above. If no relevant information exists, refer to the "Flow of Chat" for context to create an informed and
        relevant response.
        ### IF USER QUERIES IN BANGLA RESPONSE GIVE IN BANGLA ELSE ENGLISH ###

        Flow of Chat: {previous_responses}

        User message: {user_question}

        (NO PREAMBLE)
        """


@app.post("/debug/raw")
def debug_raw(req: ChatRequest):
    """Report what the model actually returned, not just its content."""
    from langchain_core.prompts import PromptTemplate

    retrieved = chain.collection.query(
        query_texts=req.query, n_results=N_RESULTS
    ).get("documents")
    retrieved_chars = len(str(retrieved))

    prompt = PromptTemplate.from_template(_DEBUG_TEMPLATE)
    runnable = prompt | chain.llm

    try:
        res = runnable.invoke(
            input={
                "user_question": req.query,
                "answer": retrieved,
                "previous_responses": "",
            }
        )
    except Exception as exc:  # noqa: BLE001
        return {"raised": str(exc)[:600]}

    extra = getattr(res, "additional_kwargs", {}) or {}
    meta = getattr(res, "response_metadata", {}) or {}

    # Reasoning models can return their working here instead of in content.
    reasoning = extra.get("reasoning") or ""

    return {
        "retrieved_chars": retrieved_chars,
        "content_len": len(res.content or ""),
        "content_head": (res.content or "")[:200],
        "finish_reason": meta.get("finish_reason"),
        "token_usage": meta.get("token_usage"),
        "model_name": meta.get("model_name"),
        "usage_metadata": getattr(res, "usage_metadata", None),
        "additional_kwargs_keys": sorted(extra.keys()),
        "reasoning_len": len(reasoning),
        "reasoning_head": reasoning[:400],
    }
