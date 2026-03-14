import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
import shutil
import datetime

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente GCBA",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Fondo principal */
.stApp {
    background-color: #f5f4f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #003087;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8f0fe !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}

/* Header */
.gcba-header {
    background: #003087;
    color: white;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.gcba-header h1 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.gcba-header p {
    margin: 0;
    font-size: 0.8rem;
    opacity: 0.75;
    font-weight: 300;
}

/* Mensajes del chat */
.chat-message-user {
    background: #003087;
    color: white;
    padding: 0.85rem 1.1rem;
    border-radius: 12px 12px 4px 12px;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.92rem;
    line-height: 1.5;
}
.chat-message-bot {
    background: white;
    color: #1a1a1a;
    padding: 0.85rem 1.1rem;
    border-radius: 12px 12px 12px 4px;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid #e0ddd5;
}
.chat-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.6;
    color: #444;
}
.chat-label-user {
    text-align: right;
    color: #003087;
}

/* Badge de estado */
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.status-ok { background: #d4edda; color: #155724; }
.status-warn { background: #fff3cd; color: #856404; }

/* Input */
.stChatInput input {
    border-radius: 10px !important;
    border: 1.5px solid #c8c5bc !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Botones */
.stButton button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    border: none !important;
}

/* Fuentes en sidebar */
.stFileUploader label {
    color: #e8f0fe !important;
}

/* Separador */
hr { border-color: #e0ddd5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ───────────────────────────────────────────────────────────────
DOCS_DIR = Path("documentos")
CHROMA_DIR = Path("chroma_db")
DOCS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Funciones de utilidad ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_vectorstore():
    embeddings = get_embeddings()
    if not any(CHROMA_DIR.iterdir()) if CHROMA_DIR.exists() else True:
        return None
    try:
        vs = FAISS.load_local(str(CHROMA_DIR), embeddings, allow_dangerous_deserialization=True)
        if vs._collection.count() == 0:
            return None
        return vs
    except Exception:
        return None

def rebuild_vectorstore():
    """Recarga todos los documentos y reconstruye ChromaDB."""
    embeddings = get_embeddings()
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    for filepath in DOCS_DIR.rglob("*"):
        if filepath.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                # Agregar metadato de fuente legible
                for c in chunks:
                    c.metadata["source_name"] = filepath.name
                all_docs.extend(chunks)
            except Exception as e:
                st.warning(f"No se pudo cargar {filepath.name}: {e}")
        elif filepath.suffix.lower() in [".txt", ".md"]:
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                for c in chunks:
                    c.metadata["source_name"] = filepath.name
                all_docs.extend(chunks)
            except Exception as e:
                st.warning(f"No se pudo cargar {filepath.name}: {e}")

    if not all_docs:
        return None

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir()

    vs = FAISS.from_documents(all_docs, embeddings)
    vs.save_local(str(CHROMA_DIR))
    return vs

def build_chain(vectorstore, api_key: str):
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
    )
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain

SYSTEM_PROMPT_PREFIX = """Eres el asistente oficial del Gobierno de la Ciudad de Buenos Aires (GCBA).
Tu función es ayudar a los empleados a consultar normativas, procedimientos, organigrama y documentos internos.
Responde siempre en español rioplatense, de manera clara, precisa y profesional.
Si la información no está en los documentos disponibles, indícalo claramente en lugar de inventar.
Cita la fuente del documento cuando sea relevante.
"""

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Asistente GCBA")
    st.markdown("---")

    # API Key
    st.markdown("### Configuración")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Obtenela gratis en console.groq.com",
        placeholder="gsk_...",
    )

    st.markdown("---")

    # Panel de administración
    st.markdown("### Panel de administración")
    st.markdown("**Subir documentos**")
    uploaded_files = st.file_uploader(
        "PDF, TXT o MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            dest = DOCS_DIR / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
        st.success(f"{len(uploaded_files)} archivo(s) guardado(s)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Indexar docs", use_container_width=True):
            with st.spinner("Indexando..."):
                vs = rebuild_vectorstore()
                if vs:
                    st.session_state["vectorstore"] = vs
                    st.session_state["chain"] = None
                    st.success("Listo")
                else:
                    st.error("Sin documentos")
    with col2:
        if st.button("Nueva conv.", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["chain"] = None
            st.rerun()

    st.markdown("---")

    # Documentos actuales
    docs_list = list(DOCS_DIR.glob("*"))
    st.markdown(f"**Documentos cargados ({len(docs_list)})**")
    if docs_list:
        for d in docs_list:
            col_d, col_x = st.columns([4, 1])
            with col_d:
                st.markdown(f"<small>📄 {d.name}</small>", unsafe_allow_html=True)
            with col_x:
                if st.button("✕", key=f"del_{d.name}"):
                    d.unlink()
                    st.rerun()
    else:
        st.markdown("<small>Ninguno todavía</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small>v1.0 · Gobierno CABA</small>", unsafe_allow_html=True)

# ── Área principal ───────────────────────────────────────────────────────────
st.markdown("""
<div class="gcba-header">
    <div>
        <h1>🏛️ Asistente de Consultas Internas</h1>
        <p>Gobierno de la Ciudad Autónoma de Buenos Aires · Consultas sobre normativas, organigrama y procedimientos</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Estado de la base de conocimiento
vs_status = load_vectorstore()
if vs_status:
    n = vs_status._collection.count()
    st.markdown(f'<span class="status-badge status-ok">Base activa · {n} fragmentos indexados</span>', unsafe_allow_html=True)
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = vs_status
else:
    st.markdown('<span class="status-badge status-warn">Sin base de conocimiento · Subí documentos en el panel lateral</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar historial
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-label chat-label-user">Vos</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-label">Asistente GCBA</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("Ver fuentes utilizadas"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['name']}** — *{src['snippet']}*")

# ── Input del usuario ────────────────────────────────────────────────────────
if prompt := st.chat_input("Consultá normativas, organigrama, procedimientos..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    if not api_key:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Para responder necesito una Groq API Key. Configurala en el panel lateral.",
            "sources": [],
        })
        st.rerun()

    vs = st.session_state.get("vectorstore")
    if not vs:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Todavía no hay documentos indexados. Subí los PDFs o textos en el panel lateral y hacé clic en 'Indexar docs'.",
            "sources": [],
        })
        st.rerun()

    # Construir chain si no existe o cambió la API key
    if "chain" not in st.session_state or not st.session_state["chain"]:
        st.session_state["chain"] = build_chain(vs, api_key)

    chain = st.session_state["chain"]

    with st.spinner("Consultando base de conocimiento..."):
        try:
            full_prompt = f"{SYSTEM_PROMPT_PREFIX}\n\nPregunta del empleado: {prompt}"
            result = chain({"question": full_prompt})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])

            # Extraer fuentes únicas
            sources = []
            seen = set()
            for doc in source_docs:
                name = doc.metadata.get("source_name", doc.metadata.get("source", "Documento"))
                if name not in seen:
                    seen.add(name)
                    snippet = doc.page_content[:120].replace("\n", " ").strip()
                    sources.append({"name": name, "snippet": snippet + "..."})

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as e:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Ocurrió un error al procesar la consulta: {str(e)}",
                "sources": [],
            })

    st.rerun()
