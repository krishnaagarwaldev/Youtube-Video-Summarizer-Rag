import os
import hashlib
import threading
import shelve
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
CACHE_DIR            = ".rag_cache"
MEMORY_K             = 6
MAX_TRANSCRIPT_CHARS = 200_000

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🎬 YouTube RAG Chatbot")
st.caption("Ask questions about any YouTube video using its transcript · Memory + Cache enabled")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "chain_core":      None,
    "llm":             None,
    "retriever":       None,
    "msg_history":     None,
    "chat_history":    [],
    "loaded_video_id": None,
    "suggested_qs":    [],
    "transcript_lang": "en",
    "transcript_src":  None,   # "youtube" | "upload"
    "cache_hits":      0,
    "cache_misses":    0,
    "model_repo":      list(MODELS.values())[0],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Thread-safe LLM cache ──────────────────────────────────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)
_cache_lock = threading.Lock()

def _cache_key(model_repo: str, question: str, context: str, history_msgs: list) -> str:
    history_str = str([(m.type, m.content[:100]) for m in history_msgs])
    raw = f"{model_repo}||{question.strip().lower()}||{context[:500]}||{history_str}"
    return hashlib.sha256(raw.encode()).hexdigest()

def cache_get(key: str) -> Optional[str]:
    try:
        with _cache_lock:
            with shelve.open(os.path.join(CACHE_DIR, "llm_cache")) as db:
                return db.get(key)
    except Exception:
        return None

def cache_set(key: str, value: str) -> None:
    try:
        with _cache_lock:
            with shelve.open(os.path.join(CACHE_DIR, "llm_cache")) as db:
                db[key] = value
    except Exception:
        pass  # non-fatal

def cache_clear(vid_prefix: Optional[str] = None) -> None:
    try:
        with _cache_lock:
            with shelve.open(os.path.join(CACHE_DIR, "llm_cache")) as db:
                if vid_prefix is None:
                    db.clear()
                else:
                    for k in [k for k in db.keys() if k.startswith(vid_prefix)]:
                        del db[k]
    except Exception:
        pass

# ── Cached heavy resources ─────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(model_repo: str) -> ChatHuggingFace:
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

def fetch_transcript(vid: str) -> tuple:
    """
    Returns (transcript_text, lang_code, optional_note).
    Raises RuntimeError with a clean, user-friendly message on any failure.
    """
    try:
        api = YouTubeTranscriptApi()
    except Exception as e:
        raise RuntimeError(f"Failed to initialise transcript client: {e}")

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
    except Exception:
        raise RuntimeError(
            "YouTube blocked this server's IP address — this is common on cloud platforms. "
            "Please use the **Upload .txt file** option instead."
        )

    if not available:
        raise RuntimeError(
            "No transcripts are available for this video. "
            "Try the **Upload .txt file** option instead."
        )

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

    # Last resort: auto-translate
    for t in available:
        try:
            tl   = t.translate("en").fetch()
            text = " ".join(c["text"] if isinstance(c, dict) else c.text for c in tl)
            return text, "en-translated", (
                f"Auto-translated from **{t.language}** to English. Quality may vary."
            )
        except Exception:
            continue

    raise RuntimeError(
        "Could not retrieve any transcript for this video. "
        "Try the **Upload .txt file** option instead."
    )

def read_uploaded_transcript(uploaded_file) -> str:
    """Read and decode an uploaded .txt file; tries UTF-8 then latin-1."""
    try:
        raw = uploaded_file.read()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
    except Exception as e:
        raise RuntimeError(f"Could not read the uploaded file: {e}")

def build_chain(llm):
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

def build_index_and_chain(transcript: str, model_repo: str):
    """Split → embed → FAISS → chain. Returns (retriever, chain, llm, n_chunks)."""
    try:
        splitter     = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks       = splitter.create_documents([transcript])
        embeddings   = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever    = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as e:
        raise RuntimeError(f"Failed to build search index: {e}")

    try:
        llm = get_llm(model_repo)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load the AI model. "
            f"Check that your HUGGINGFACEHUB_API_TOKEN is valid.\n\n`{e}`"
        )

    return retriever, build_chain(llm), llm, len(chunks)

def generate_suggested_questions(retriever, llm) -> list:
    try:
        sample_docs = retriever.invoke("main topic summary overview")
        sample_text = "\n".join(d.page_content for d in sample_docs[:2])
        msg = HumanMessage(content=(
            "Based on the following transcript excerpt, generate exactly 4 short, "
            "interesting questions a viewer might ask. Return ONLY a numbered list, "
            "one question per line, no extra commentary.\n\n" + sample_text[:1500]
        ))
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

def reset_session():
    for k in ["chain_core", "llm", "loaded_video_id", "retriever", "msg_history"]:
        st.session_state[k] = None
    st.session_state.chat_history    = []
    st.session_state.suggested_qs   = []
    st.session_state.transcript_lang = "en"
    st.session_state.transcript_src  = None
    st.session_state.cache_hits      = 0
    st.session_state.cache_misses    = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    model_label = st.selectbox("🤖 Model", list(MODELS.keys()))
    model_repo  = MODELS[model_label]
    st.session_state.model_repo = model_repo

    st.divider()

    source_mode = st.radio(
        "📄 Transcript Source",
        ["🔗 YouTube URL / ID", "📁 Upload .txt file"],
        horizontal=True,
    )

    transcript      = None
    vid             = None
    transcript_lang = "en"
    transcript_note = None
    do_build        = False   # gate that controls whether build pipeline runs

    # ── Mode A: YouTube ────────────────────────────────────────────────────────
    if source_mode == "🔗 YouTube URL / ID":
        video_input = st.text_input(
            "YouTube Video ID or URL",
            placeholder="e.g. Gfr50f6ZBvo  or  https://youtu.be/Gfr50f6ZBvo",
        )
        load_btn = st.button("🚀 Load Video & Build Index", use_container_width=True)

        if load_btn:
            if not HF_TOKEN:
                st.error("🔑 Add **HUGGINGFACEHUB_API_TOKEN** to your .env file and restart the app.")
            elif not video_input.strip():
                st.warning("Please enter a YouTube video ID or URL.")
            else:
                vid = extract_video_id(video_input)
                with st.spinner("📥 Fetching transcript from YouTube…"):
                    try:
                        transcript, transcript_lang, transcript_note = fetch_transcript(vid)
                        do_build = True
                    except TranscriptsDisabled:
                        st.error(
                            "🚫 **Transcripts are disabled** for this video.\n\n"
                            "Switch to **Upload .txt file** and paste the transcript manually."
                        )
                    except RuntimeError as e:
                        st.error(f"⚠️ {e}")
                    except Exception as e:
                        st.error(
                            "❌ An unexpected error occurred while fetching the transcript. "
                            "Try the **Upload .txt file** option instead."
                        )

    # ── Mode B: File upload ────────────────────────────────────────────────────
    else:
        st.caption(
            "💡 To get a transcript: open the video on YouTube → click **⋯** → "
            "**Open transcript** → copy all text → paste into a .txt file."
        )
        uploaded_file = st.file_uploader(
            "Upload transcript (.txt)",
            type=["txt"],
            help="Plain text file containing the video transcript.",
        )
        video_id_input = st.text_input(
            "Video ID (optional — enables the preview player)",
            placeholder="e.g. Gfr50f6ZBvo",
        )
        load_btn = st.button("🚀 Build Index from File", use_container_width=True)

        if load_btn:
            if not HF_TOKEN:
                st.error("🔑 Add **HUGGINGFACEHUB_API_TOKEN** to your .env file and restart the app.")
            elif uploaded_file is None:
                st.warning("Please upload a .txt transcript file first.")
            else:
                vid = extract_video_id(video_id_input) if video_id_input.strip() else "uploaded"
                try:
                    transcript      = read_uploaded_transcript(uploaded_file)
                    transcript_lang = "uploaded"
                    transcript_note = f"Loaded from **{uploaded_file.name}**."
                    do_build        = True
                except RuntimeError as e:
                    st.error(f"⚠️ {e}")

    # ── Shared build pipeline ──────────────────────────────────────────────────
    if do_build and transcript is not None:
        if transcript_note:
            st.info(f"ℹ️ {transcript_note}")

        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            st.warning(
                f"⚠️ Transcript is very long ({len(transcript):,} chars). "
                "Embedding may be slow."
            )

        if len(transcript.strip()) < 100:
            st.error(
                "⚠️ The transcript appears to be empty or too short. "
                "Please check the file and try again."
            )
        else:
            with st.spinner("✂️ Splitting & embedding…"):
                try:
                    retriever, chain_core, llm, n_chunks = build_index_and_chain(transcript, model_repo)
                except RuntimeError as e:
                    st.error(f"⚠️ {e}")
                    retriever = None

            if retriever is not None:
                st.session_state.chain_core      = chain_core
                st.session_state.llm             = llm
                st.session_state.retriever       = retriever
                st.session_state.msg_history     = ChatMessageHistory()
                st.session_state.loaded_video_id = vid
                st.session_state.transcript_lang = transcript_lang
                st.session_state.transcript_src  = (
                    "upload" if source_mode != "🔗 YouTube URL / ID" else "youtube"
                )
                st.session_state.chat_history    = []
                st.session_state.cache_hits      = 0
                st.session_state.cache_misses    = 0

                with st.spinner("💡 Generating suggested questions…"):
                    st.session_state.suggested_qs = generate_suggested_questions(retriever, llm)

                st.success(f"✅ Ready! {n_chunks} chunks · {model_label}")

    # ── Stats + controls ───────────────────────────────────────────────────────
    if st.session_state.loaded_video_id:
        st.divider()
        src_icon = "📁" if st.session_state.transcript_src == "upload" else "▶️"
        st.markdown(f"**{src_icon} Loaded:** `{st.session_state.loaded_video_id}`")
        lang = st.session_state.get("transcript_lang", "")
        if lang:
            st.caption(f"Transcript: `{lang}`")

        mem_count = len(st.session_state.msg_history.messages) // 2 if st.session_state.msg_history else 0
        st.caption(f"🧠 Memory: **{mem_count}** turn(s) · window = {MEMORY_K}")

        hits, misses = st.session_state.cache_hits, st.session_state.cache_misses
        total = hits + misses
        rate  = f"{hits/total*100:.0f}%" if total else "—"
        st.caption(f"⚡ Cache: **{hits}** hits / **{misses}** misses ({rate})")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.msg_history  = ChatMessageHistory()
                st.rerun()
        with col2:
            if st.button("🗄️ Clear cache", use_container_width=True):
                cache_clear(vid_prefix=st.session_state.loaded_video_id)
                st.session_state.cache_hits   = 0
                st.session_state.cache_misses = 0
                st.toast("Cache cleared!")

        if st.button("🔄 Load different video", use_container_width=True):
            reset_session()
            st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
if st.session_state.chain_core is None:
    st.info(
        "👈 Enter a YouTube URL **or** upload a transcript .txt file in the sidebar, "
        "then click **Load**."
    )
    st.stop()

vid        = st.session_state.loaded_video_id
model_repo = st.session_state.model_repo

col_video, col_chat = st.columns([1, 1.4], gap="large")

with col_video:
    if st.session_state.transcript_src == "youtube" and vid != "uploaded":
        st.subheader("📺 Video Preview")
        try:
            st.components.v1.iframe(
                src=f"https://www.youtube.com/embed/{vid}",
                height=315,
                scrolling=False,
            )
        except Exception:
            st.caption("Could not load video preview.")
    else:
        st.subheader("📄 Uploaded Transcript")
        st.info(
            "No video preview for manually uploaded transcripts. "
            "If you have a video ID, re-load and enter it in the optional field."
        )

    if st.session_state.suggested_qs:
        st.subheader("💡 Suggested Questions")
        st.caption("Click to ask instantly.")
        for q in st.session_state.suggested_qs:
            if st.button(q, key=f"sq_{q}", use_container_width=True):
                st.session_state["_pending_question"] = q
                st.rerun()

with col_chat:
    st.subheader("💬 Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("cached"):
                st.caption("⚡ cached")
            st.markdown(msg["content"])

    pending  = st.session_state.get("_pending_question")
    question = st.chat_input("Ask something about the video…") or pending
    if pending and question == pending:
        del st.session_state["_pending_question"]

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        answer = ""  # safe default

        try:
            retrieved_docs = st.session_state.retriever.invoke(question)
            context_str    = format_docs(retrieved_docs)
        except Exception as e:
            st.error(f"⚠️ Search failed — could not retrieve relevant chunks: `{e}`")
            st.stop()

        history_msgs  = get_trimmed_history(st.session_state.msg_history)
        ck            = vid + "_" + _cache_key(model_repo, question, context_str, history_msgs)
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
                try:
                    def token_stream(chain, ctx, q, hist):
                        yield from chain.stream({"context": ctx, "question": q, "history": hist})

                    answer = st.write_stream(
                        token_stream(st.session_state.chain_core, context_str, question, history_msgs)
                    )
                    cache_set(ck, answer)
                    st.session_state.msg_history.add_user_message(question)
                    st.session_state.msg_history.add_ai_message(answer)
                except Exception as e:
                    st.error(
                        "⚠️ The AI model failed to respond. This is usually a temporary API issue "
                        "— please try again in a moment."
                    )
                    st.stop()

        if answer:
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": answer,
                "cached":  cached_answer is not None,
            })
        st.rerun()