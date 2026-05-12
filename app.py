import os
import hashlib
import threading
import shelve
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS = {
    "⚡ Llama 3.1 8B Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "🌟 Qwen 2.5 7B Instruct":  "Qwen/Qwen2.5-7B-Instruct",
}
CACHE_DIR  = ".rag_cache"
MEMORY_K   = 6
MAX_TRANSCRIPT_CHARS = 200_000   # ~3 h of speech; warn beyond this

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",   # FIX #14
)
st.title("🎬 YouTube RAG Chatbot")
st.caption("Ask questions about any YouTube video using its transcript · Memory + Cache enabled")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "chain_core": None,
    "llm": None,
    "retriever": None,
    "msg_history": None,
    "chat_history": [],
    "loaded_video_id": None,
    "suggested_qs": [],
    "transcript_lang": "en",
    "cache_hits": 0,
    "cache_misses": 0,
    "model_repo": list(MODELS.values())[0],   # FIX #1 — always available at module level
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Thread-safe LLM cache (FIX #2) ────────────────────────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)
_cache_lock = threading.Lock()

def _cache_key(model_repo: str, question: str, context: str, history_msgs: list) -> str:
    """FIX #7 — include recent history in key so follow-up Qs aren't mis-served."""
    history_str = str([(m.type, m.content[:100]) for m in history_msgs])
    raw = f"{model_repo}||{question.strip().lower()}||{context[:500]}||{history_str}"
    return hashlib.sha256(raw.encode()).hexdigest()

def cache_get(key: str):
    with _cache_lock:                                         # FIX #2
        with shelve.open(os.path.join(CACHE_DIR, "llm_cache")) as db:
            return db.get(key)

def cache_set(key: str, value: str):
    with _cache_lock:                                         # FIX #2
        with shelve.open(os.path.join(CACHE_DIR, "llm_cache")) as db:
            db[key] = value

def cache_clear(vid_prefix: str | None = None):
    """FIX #16 — optionally clear only entries for a specific video."""
    with _cache_lock:
        with shelve.open(os.path.join(CACHE_DIR, "llm_cache")) as db:
            if vid_prefix is None:
                db.clear()
            else:
                keys_to_delete = [k for k in db.keys() if k.startswith(vid_prefix)]
                for k in keys_to_delete:
                    del db[k]

# ── Cached heavy resources (FIX #4, #5) ───────────────────────────────────────
@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Loaded once per process; reused across all video loads."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(model_repo: str) -> ChatHuggingFace:
    """One LLM instance per model repo; survives across reruns."""
    endpoint = HuggingFaceEndpoint(
        repo_id=model_repo,
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_video_id(raw: str) -> str:
    raw = raw.strip()
    if "youtu.be/" in raw:
        return raw.split("youtu.be/")[-1].split("?")[0]
    if "v=" in raw:
        return raw.split("v=")[-1].split("&")[0]
    return raw

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def fetch_transcript(vid: str):
    api = YouTubeTranscriptApi()
    priority_langs = ["en", "hi"]
    for lang in priority_langs:
        try:
            tl   = api.fetch(vid, languages=[lang])
            text = " ".join(c["text"] if isinstance(c, dict) else c.text for c in tl)
            return text, lang, None
        except Exception:
            continue
    try:
        available = list(api.list(vid))
    except Exception as e:
        raise RuntimeError(f"Could not list transcripts: {e}")
    if not available:
        raise RuntimeError("No transcripts available for this video.")
    for t in available:
        lc = t.language_code
        if lc in priority_langs:
            continue
        try:
            tl   = api.fetch(vid, languages=[lc])
            text = " ".join(c["text"] if isinstance(c, dict) else c.text for c in tl)
            return text, lc, f"No EN/HI transcript — using **{t.language} ({lc})**."
        except Exception:
            continue
    # FIX #10 — warn user before attempting auto-translation
    st.warning("⚠️ No direct transcript found. Attempting auto-translation — quality may be reduced.")
    for t in available:
        try:
            tl   = t.translate("en").fetch()
            text = " ".join(c["text"] if isinstance(c, dict) else c.text for c in tl)
            return text, "en-translated", f"Auto-translated from **{t.language}** to English."
        except Exception:
            continue
    raise RuntimeError("Could not fetch any transcript for this video.")

def build_chain(llm):
    """
    FIX #11 — retrieval is fully decoupled from the chain.
    The chain only handles: context + history + question → LLM → string.
    Retrieval is done by the caller which passes `context` explicitly.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant answering questions about a YouTube video transcript.
Answer ONLY from the provided transcript context and the conversation history below.
If the context is insufficient, say you don't know.
Use Markdown for formatting (**bold**, _italic_, lists, `code`).
Use LaTeX for math: inline $...$ and block $$...$$.\n\nTranscript Context:\n{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

def generate_suggested_questions(retriever, llm) -> list:
    sample_docs = retriever.invoke("main topic summary overview")
    sample_text = "\n".join(d.page_content for d in sample_docs[:2])
    msg = HumanMessage(content=(
        "Based on the following transcript excerpt, generate exactly 4 short, "
        "interesting questions a viewer might ask. Return ONLY a numbered list, "
        "one question per line, no extra commentary.\n\n" + sample_text[:1500]
    ))
    try:
        raw   = llm.invoke([msg]).content
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        qs    = [l.lstrip("1234567890.)- ").strip() for l in lines if "?" in l]
        if qs:
            return qs[:4]
    except Exception:
        pass
    return [
        "What is the main topic of this video?",
        "Summarize the video in 3 bullet points.",
        "What are the key takeaways?",
        "Are there any surprising facts mentioned?",
    ]

def get_trimmed_history(msg_history: ChatMessageHistory) -> list:
    msgs = msg_history.messages
    return msgs[-(MEMORY_K * 2):] if len(msgs) > MEMORY_K * 2 else msgs

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    model_label = st.selectbox("🤖 Model", list(MODELS.keys()))
    model_repo  = MODELS[model_label]
    # FIX #1 — keep model_repo in session state so it's safe to read in main area
    st.session_state.model_repo = model_repo

    video_input = st.text_input(
        "YouTube Video ID or URL",
        placeholder="e.g. Gfr50f6ZBvo  or  https://youtu.be/Gfr50f6ZBvo",
    )

    load_btn = st.button("🚀 Load Video & Build Index", use_container_width=True)

    if load_btn:
        if not HF_TOKEN:
            st.error("Add HUGGINGFACEHUB_API_TOKEN to your .env and restart.")
        elif not video_input.strip():
            st.error("Please enter a YouTube video ID or URL.")
        else:
            vid = extract_video_id(video_input)

            with st.spinner("📥 Fetching transcript…"):
                try:
                    transcript, lang_used, lang_note = fetch_transcript(vid)
                    st.session_state.transcript_lang = lang_used
                    if lang_note:
                        st.info(f"ℹ️ {lang_note}")
                except TranscriptsDisabled:
                    st.error("Transcripts are disabled for this video.")
                    st.stop()
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            # FIX #17 — warn on very long transcripts
            if len(transcript) > MAX_TRANSCRIPT_CHARS:
                st.warning(
                    f"⚠️ Transcript is very long ({len(transcript):,} chars). "
                    "Embedding may be slow and context may be truncated."
                )

            with st.spinner("✂️ Splitting & embedding…"):
                splitter     = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks       = splitter.create_documents([transcript])
                embeddings   = get_embeddings()   # FIX #4 — cached
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever    = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            with st.spinner(f"🤖 Loading {model_label}…"):
                try:
                    llm = get_llm(model_repo)   # FIX #5 — cached
                except Exception as e:
                    st.error(f"LLM init failed: {e}")
                    st.stop()

            chain_core  = build_chain(llm)        # FIX #11 — decoupled
            msg_history = ChatMessageHistory()

            st.session_state.chain_core      = chain_core
            st.session_state.llm             = llm
            st.session_state.retriever       = retriever
            st.session_state.msg_history     = msg_history
            st.session_state.loaded_video_id = vid
            st.session_state.chat_history    = []
            st.session_state.cache_hits      = 0
            st.session_state.cache_misses    = 0

            # FIX #15 — only generate if not already set for this video
            with st.spinner("💡 Generating suggested questions…"):
                st.session_state.suggested_qs = generate_suggested_questions(retriever, llm)

            st.success(f"✅ Ready! {len(chunks)} chunks · {model_label}")

    # ── Stats + controls ───────────────────────────────────────────────────────
    if st.session_state.loaded_video_id:
        st.divider()
        st.markdown(f"**Video:** `{st.session_state.loaded_video_id}`")
        lang = st.session_state.get("transcript_lang", "")
        if lang:
            st.caption(f"Transcript language: `{lang}`")

        mem_count = 0
        if st.session_state.msg_history:
            mem_count = len(st.session_state.msg_history.messages) // 2
        st.caption(f"🧠 Memory: **{mem_count}** turn(s) · window = {MEMORY_K}")

        hits   = st.session_state.cache_hits
        misses = st.session_state.cache_misses
        total  = hits + misses
        rate   = f"{hits/total*100:.0f}%" if total else "—"
        st.caption(f"⚡ Cache: **{hits}** hits / **{misses}** misses ({rate})")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.msg_history  = ChatMessageHistory()
                st.rerun()
        with col2:
            if st.button("🗄️ Clear cache", use_container_width=True):
                # FIX #16 — clear only this video's cache entries
                cache_clear(vid_prefix=st.session_state.loaded_video_id)
                st.session_state.cache_hits   = 0
                st.session_state.cache_misses = 0
                st.toast("Cache cleared for this video!")

        if st.button("🔄 Load different video", use_container_width=True):
            for k in ["chain_core", "llm", "loaded_video_id", "retriever", "msg_history"]:
                st.session_state[k] = None
            st.session_state.chat_history    = []
            st.session_state.suggested_qs   = []
            st.session_state.transcript_lang = "en"
            st.session_state.cache_hits      = 0
            st.session_state.cache_misses    = 0
            st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
if st.session_state.chain_core is None:
    st.info("👈 Enter a YouTube video ID/URL in the sidebar and click **Load Video & Build Index**.")
    st.stop()

vid        = st.session_state.loaded_video_id
model_repo = st.session_state.model_repo   # FIX #1 — read from session state

col_video, col_chat = st.columns([1, 1.4], gap="large")

with col_video:
    st.subheader("📺 Video Preview")
    st.components.v1.iframe(
        src=f"https://www.youtube.com/embed/{vid}",
        height=315,
        scrolling=False,
    )

    if st.session_state.suggested_qs:
        st.subheader("💡 Suggested Questions")
        st.caption("Click to ask instantly.")
        for q in st.session_state.suggested_qs:
            if st.button(q, key=f"sq_{q}", use_container_width=True):
                # FIX #3 — set pending AFTER clearing to avoid double-rerun edge case
                st.session_state["_pending_question"] = q
                st.rerun()

with col_chat:
    st.subheader("💬 Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("cached"):
                st.caption("⚡ cached")
            st.markdown(msg["content"])

    # FIX #3 — read then clear, don't pop before processing
    pending  = st.session_state.get("_pending_question")
    question = st.chat_input("Ask something about the video…") or pending
    if pending and question == pending:
        del st.session_state["_pending_question"]

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # FIX #6 — retrieve once; pass context to both cache key and chain
        retrieved_docs  = st.session_state.retriever.invoke(question)
        context_str     = format_docs(retrieved_docs)
        history_msgs    = get_trimmed_history(st.session_state.msg_history)

        # FIX #7 — history-aware cache key; FIX #16 — vid-prefixed key
        ck = vid + "_" + _cache_key(model_repo, question, context_str, history_msgs)
        cached_answer = cache_get(ck)

        with st.chat_message("assistant"):
            if cached_answer:
                st.session_state.cache_hits += 1
                st.caption("⚡ cached")
                st.markdown(cached_answer)
                answer = cached_answer
                st.session_state.msg_history.add_user_message(question)
                st.session_state.msg_history.add_ai_message(answer)
            else:
                st.session_state.cache_misses += 1

                # FIX #9 — wrap stream in try/except to handle token expiry etc.
                try:
                    def token_stream():
                        for token in st.session_state.chain_core.stream({
                            "context":  context_str,   # FIX #6 — pre-retrieved
                            "question": question,
                            "history":  history_msgs,
                        }):
                            yield token

                    answer = st.write_stream(token_stream())
                except Exception as e:
                    st.error(f"LLM call failed: {e}. Check your API token or try again.")
                    st.stop()

                cache_set(ck, answer)
                st.session_state.msg_history.add_user_message(question)
                st.session_state.msg_history.add_ai_message(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "cached": cached_answer is not None,
        })
        st.rerun()