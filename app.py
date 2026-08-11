
# =========================
# app.py - Chainlit UI for your RAG Agent (LangGraph + LanceDB + FastEmbed)
# Based on NLP_HW5_Q1.ipynb
# Run:
#   chainlit run app.py -w --port 8000
# =========================

import os
import time
import traceback
from typing import TypedDict, List, Dict, Any, Optional, Callable

import chainlit as cl

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

import lancedb
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from typing import Literal

# =========================
# Config (override with env if you want)
# =========================
# Put your lancedb_storage folder next to app.py OR set env LANCEDB_DIR to your absolute path.
LANCEDB_DIR = os.getenv("LANCEDB_DIR", "./lancedb_storage")
TABLE_NAME  = os.getenv("TABLE_NAME", "law_chunks")

# Embedding model (same as your notebook)
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-large")

# LLM model: IMPORTANT - must be supported by AvalAI
# You can override with: set OPENAI_MODEL in env
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# =========================
# Agent State (same as your notebook)
# =========================
class AgentState(TypedDict, total=False):
    user_query: str
    rewritten_query: str

    intent: str
    intent_response: str

    metadata_filter: Dict[str, Any]

    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]

    retry_count: int
    need_retry: bool

    final_answer: str

    timings: List[Dict[str, Any]]
    question_id: str
    source_law: str


# =========================
# Utility: timed wrapper (same logic as notebook)
# =========================
def timed(node_name: str, fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    def _wrapped(state: AgentState) -> AgentState:
        if "timings" not in state:
            state["timings"] = []

        start = time.perf_counter()
        out = fn(state)
        end = time.perf_counter()

        state["timings"].append({
            "node": node_name,
            "start": start,
            "end": end,
            "duration_ms": (end - start) * 1000.0
        })

        return out
    return _wrapped


# =========================
# LLM (AvalAI / OpenAI-compatible)
# =========================
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is not set (should end with /v1)")

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )


# =========================
# Embedding (FastEmbed) - same as notebook
# =========================
embedder = TextEmbedding(model_name=EMB_MODEL_NAME)

def embed_query(text: str) -> List[float]:
    return list(embedder.embed([f"query: {text}"]))[0].tolist()

def embed_passage(text: str) -> List[float]:
    return list(embedder.embed([f"passage: {text}"]))[0].tolist()


# =========================
# LanceDB connect
# =========================
db = lancedb.connect(LANCEDB_DIR)
table = db.open_table(TABLE_NAME)


# =========================
# Node 1: Rewrite Query (same intent as notebook)
# =========================
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "تو فقط وظیفه بازنویسی پرسش کاربر را داری.\n"
        "سؤال را شفاف، دقیق و مناسب بازیابی اسناد قانونی کن.\n"
        "از خودت پاسخ نده، توضیح نده، فقط متن سؤال بازنویسی‌شده را برگردان."
    ),
    ("human", "{question}")
])

def rewrite_query_node(state: AgentState) -> AgentState:
    user_q = state["user_query"]
    llm = get_llm()
    rewritten = llm.invoke(
        rewrite_prompt.format_messages(question=user_q)
    ).content.strip()
    return {"rewritten_query": rewritten}


# =========================
# Node 2: Intent Classification (Greeting / Abusive / Law)
# =========================
intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "نوع پیام کاربر را دقیقاً در یکی از این سه دسته تشخیص بده.\n"
        "فقط یکی از این سه کلمه را خروجی بده و هیچ توضیح اضافه نده:\n"
        "Greeting\nAbusive\nLaw"
    ),
    ("human", "{question}")
])

def classify_intent_node(state: AgentState) -> AgentState:
    user_q = state["user_query"]
    llm = get_llm()

    label = llm.invoke(
        intent_prompt.format_messages(question=user_q)
    ).content.strip()

    if label == "Greeting":
        return {
            "intent": "greeting",
            "intent_response": "سلام 👋 خوش اومدی! سؤال حقوقی‌ات رو بپرس تا بررسی کنم."
        }

    if label == "Abusive":
        return {
            "intent": "abusive",
            "intent_response": (
                "من برای کمک طراحی شدم و وارد گفت‌وگوی توهین‌آمیز نمی‌شوم. "
                "اگر سوالت را محترمانه مطرح کنی، با کمال میل راهنمایی می‌کنم."
            )
        }

    return {"intent": "law"}


# =========================
# Node 3: Extract Metadata (structured output)
# Based on your fixed allowed titles/domains
# =========================
LAW_TITLES = Literal[
    "قانون تأمين اجتماعي",
    "قانون ثبت اسناد و املاك",
    "قانون روابط موجر و مستأجر",
    "قانون صدور چك",
    "قانون كار",
    "قانون ماليات بر ارزش افزوده",
    "قانون مبارزه با قاچاق كالا و ارز",
]

LAW_DOMAINS = Literal[
    "anti_smuggling",
    "check",
    "labor",
    "landlord_tenant",
    "registration_realestate",
    "social_security",
    "vat_tax",
]

class MetadataFilter(BaseModel):
    law_title: Optional[LAW_TITLES] = Field(default=None, description="نام قانون فقط از بین مقادیر مجاز")
    law_domain: Optional[LAW_DOMAINS] = Field(default=None, description="دامنه فقط از بین مقادیر مجاز")

    # Optional: if your DB has these fields, we can use them; otherwise they will be ignored safely
    article_no: Optional[str] = Field(default=None, description="شماره ماده (اگر قابل استخراج بود)")
    has_tabserah: Optional[bool] = Field(default=None, description="آیا تبصره ذکر شده است یا خیر")

metadata_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "از روی پرسش بازنویسی‌شده، اگر ممکن بود فیلترهای متادیتایی استخراج کن.\n"
        "قانون (law_title) فقط باید یکی از این ۷ مقدار باشد:\n"
        "- قانون تأمين اجتماعي\n- قانون ثبت اسناد و املاك\n- قانون روابط موجر و مستأجر\n- قانون صدور چك\n- قانون كار\n- قانون ماليات بر ارزش افزوده\n- قانون مبارزه با قاچاق كالا و ارز\n"
        "law_domain هم فقط یکی از این‌ها باشد:\n"
        "anti_smuggling, check, labor, landlord_tenant, registration_realestate, social_security, vat_tax\n"
        "اگر مطمئن نیستی، مقدار را null بگذار."
    ),
    ("human", "{question}")
])

def extract_metadata_node(state: AgentState) -> AgentState:
    rewritten_q = state["rewritten_query"]
    llm = get_llm()
    extractor = llm.with_structured_output(MetadataFilter)
    metadata: MetadataFilter = extractor.invoke(
        metadata_prompt.format_messages(question=rewritten_q)
    )
    return {"metadata_filter": metadata.model_dump(exclude_none=True)}


# =========================
# Node 4: Context Retrieve (LanceDB vector search + where)
# Same logic style as notebook (embedding then table.search(embedding))
# =========================
def build_where_clause(metadata_filter: dict) -> str:
    clauses = []

    # Apply only if those columns exist in your DB schema
    schema_fields = set(table.schema.names)

    if "law_title" in metadata_filter and "law_title" in schema_fields:
        clauses.append(f"law_title = '{metadata_filter['law_title']}'")

    if "law_domain" in metadata_filter and "law_domain" in schema_fields:
        clauses.append(f"law_domain = '{metadata_filter['law_domain']}'")

    if "article_no" in metadata_filter and "article_no" in schema_fields:
        clauses.append(f"article_no = '{metadata_filter['article_no']}'")

    if "has_tabserah" in metadata_filter and "has_tabserah" in schema_fields:
        val = "true" if metadata_filter["has_tabserah"] else "false"
        clauses.append(f"has_tabserah = {val}")

    return " AND ".join(clauses)

def context_retrieve_node(state: AgentState, K: int = 10) -> AgentState:
    query = state["rewritten_query"]
    metadata_filter = state.get("metadata_filter", {}) or {}

    query_embedding = embed_query(query)

    where_clause = build_where_clause(metadata_filter)

    search = table.search(query_embedding)
    if where_clause:
        search = search.where(where_clause)

    results = search.limit(K).to_list()
    return {"retrieved_docs": results}


# =========================
# Node 5: Rerank (same as notebook; score/_distance)
# =========================
def rerank_node(state: AgentState, N: int = 3, min_score: float = 0.15) -> AgentState:
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    def get_score(doc):
        if "score" in doc:
            return float(doc["score"])
        if "_distance" in doc:
            return 1.0 / (1.0 + float(doc["_distance"]))
        return 0.0

    docs_sorted = sorted(docs, key=get_score, reverse=True)
    top_docs = docs_sorted[:N]

    best_score = get_score(top_docs[0]) if top_docs else 0.0
    relevance_ok = best_score >= min_score

    need_retry = (not relevance_ok) and (retry_count == 0)

    return {
        "reranked_docs": top_docs,
        "need_retry": need_retry
    }


# =========================
# Node 6: Generate Answer (same spirit as notebook)
# =========================
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "تو یک دستیار حقوقی هستی که فقط و فقط بر اساس متن قوانین ارائه‌شده پاسخ می‌دهد.\n"
        "اگر اطلاعات کافی در متن‌های داده‌شده وجود نداشت، صریحاً بگو «اطلاعات کافی در اسناد بازیابی‌شده وجود ندارد».\n"
        "از خودت قانون، ماده یا تبصره اختراع نکن.\n"
        "در صورت امکان، در پاسخ نام قانون و شماره ماده را ذکر کن."
    ),
    (
        "human",
        "سؤال:\n{question}\n\n"
        "متن‌های قانونی بازیابی‌شده:\n{context}\n\n"
        "پاسخ نهایی:"
    )
])

def format_context_for_answer(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return ""

    parts = []
    for i, d in enumerate(docs, 1):
        law_title = d.get("law_title", "")
        article_no = d.get("article_no", "")
        text = d.get("text", "")

        header = f"[{i}]"
        if law_title:
            header += f" {law_title}"
        if article_no:
            header += f" | ماده/بخش: {article_no}"

        parts.append(f"{header}\n{text}".strip())

    return "\n\n---\n\n".join(parts)

def generate_answer_node(state: AgentState) -> AgentState:
    question = state["rewritten_query"]
    docs = state.get("reranked_docs", [])

    context_text = format_context_for_answer(docs)

    llm = get_llm()
    answer = llm.invoke(
        answer_prompt.format_messages(
            question=question,
            context=context_text
        )
    ).content.strip()

    return {"final_answer": answer}


# =========================
# Routers + helpers (same logic as notebook)
# =========================
def end_intent_node(state: AgentState) -> AgentState:
    return {"final_answer": state.get("intent_response", "")}

def bump_retry_node(state: AgentState) -> AgentState:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "need_retry": False
    }

def intent_router(state: AgentState) -> str:
    intent = state.get("intent", "law")
    if intent in ("greeting", "abusive"):
        return "end_intent"
    return "extract_metadata"

def retry_router(state: AgentState) -> str:
    return "retry" if state.get("need_retry") else "generate"


# =========================
# Build TIMED graph (so you can still log timings if needed)
# =========================
rewrite_t = timed("rewrite_query", rewrite_query_node)
classify_t = timed("classify_intent", classify_intent_node)
extract_t  = timed("extract_metadata", extract_metadata_node)
retrieve_t = timed("context_retrieve", lambda s: context_retrieve_node(s, K=10))
rerank_t   = timed("rerank", lambda s: rerank_node(s, N=3, min_score=0.15))
generate_t = timed("generate_answer", generate_answer_node)
end_intent_t = timed("end_intent", end_intent_node)
bump_retry_t = timed("bump_retry", bump_retry_node)

graph = StateGraph(AgentState)

graph.add_node("rewrite_query", rewrite_t)
graph.add_node("classify_intent", classify_t)
graph.add_node("end_intent", end_intent_t)

graph.add_node("extract_metadata", extract_t)
graph.add_node("context_retrieve", retrieve_t)
graph.add_node("rerank", rerank_t)
graph.add_node("bump_retry", bump_retry_t)
graph.add_node("generate_answer", generate_t)

graph.set_entry_point("rewrite_query")
graph.add_edge("rewrite_query", "classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    intent_router,
    {
        "end_intent": "end_intent",
        "extract_metadata": "extract_metadata",
    }
)

graph.add_edge("end_intent", END)

graph.add_edge("extract_metadata", "context_retrieve")
graph.add_edge("context_retrieve", "rerank")

graph.add_conditional_edges(
    "rerank",
    retry_router,
    {
        "retry": "bump_retry",
        "generate": "generate_answer",
    }
)

graph.add_edge("bump_retry", "context_retrieve")
graph.add_edge("generate_answer", END)

rag_app = graph.compile()


# =========================
# Chainlit UI
# =========================
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "🤖 سامانه پرسش‌وپاسخ قوانین (RAG)\n\n"
            "سؤال حقوقی‌ات رو بپرس.\n"
            "اگر سؤال سلام/احوال‌پرسی باشد پاسخ کوتاه می‌گیرى؛ اگر توهین باشد پاسخ محترمانه می‌گیرى."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = (message.content or "").strip()
    if not question:
        await cl.Message(content="لطفاً یک سؤال وارد کنید.").send()
        return

    # Run graph safely (in thread) to avoid blocking the event loop
    try:
        state: AgentState = {
            "user_query": question,
            "retry_count": 0,
            "timings": []
        }

        result = await cl.make_async(rag_app.invoke)(state)

        answer = result.get("final_answer", "اطلاعات کافی در اسناد بازیابی‌شده وجود ندارد.")
        await cl.Message(content=answer).send()

        # Optional: store timings in the conversation for debugging (not shown by default)
        cl.user_session.set("last_timings", result.get("timings", []))

    except Exception:
        err = traceback.format_exc()
        # show a short message to user; full trace stays in terminal logs
        await cl.Message(
            content="خطایی رخ داد. لطفاً دوباره تلاش کنید. (جزئیات در ترمینال ثبت شد.)"
        ).send()
        print(err)

