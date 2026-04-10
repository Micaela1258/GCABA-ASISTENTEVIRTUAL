import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory

st.set_page_config(page_title="Asistente GCBA", page_icon="🏛️", layout="wide")

st.markdown("""<style>
.stApp { background-color: #f5f4f0; }
[data-testid="stSidebar"] { background-color: #003087; }
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
.gcba-header { background: #003087; color: white; padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; }
.gcba-header h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
.gcba-header p { margin: 0; font-size: 0.8rem; opacity: 0.75; }
.chat-message-user { background: #003087; color: white; padding: 0.85rem 1.1rem; border-radius: 12px 12px 4px 12px; margin: 0.5rem 0 0.5rem 3rem; }
.chat-message-bot { background: white; color: #1a1a1a; padding: 0.85rem 1.1rem; border-radius: 12px 12px 12px 4px; margin: 0.5rem 3rem 0.5rem 0; border: 1px solid #e0ddd5; }
.chat-label { font-size: 0.72rem; font-weight: 500; text-transform: uppercase; margin-bottom: 0.3rem; opacity: 0.6; color: #444; }
.chat-label-user { text-align: right; color: #003087; }
.status-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.72rem; font-weight: 500; }
.status-ok { background: #d4edda; color: #155724; }
.status-warn { background: #fff3cd; color: #856404; }
</style>""", unsafe_allow_html=True)

DOCS_DIR = Path(".")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

@st.cache_resource(show_spinner="Indexando documentos...")
def build_vectorstore():
    embeddings = get_embeddings()
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    for filepath in DOCS_DIR.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(filepath))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for c in chunks:
                c.metadata["source_name"] = filepath.name
            all_docs.extend(chunks)
        except Exception as e:
            st.warning(f"No se pudo cargar {filepath.name}: {e}")
    for filepath in DOCS_DIR.rglob("*.txt"):
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
    return FAISS.from_documents(all_docs, embeddings)

def build_chain(vectorstore, api_key):
    llm = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.2, max_tokens=1024)
    memory = ConversationBufferWindowMemory(k=6, memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False)

SYSTEM_PROMPT = "Eres el asistente oficial del GCBA. Ayudas a empleados a consultar normativas, procedimientos y organigrama. Responde en espanol rioplatense, claro y profesional. Si la info no esta en los documentos, decilo. Cita la fuente cuando sea relevante."

with st.sidebar:
    st.markdown("## Asistente GCBA")
    st.markdown("---")
    st.markdown("### Configuracion")
    api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    st.markdown("### Documentos cargados")
    pdfs = list(DOCS_DIR.rglob(".pdf")) + list(DOCS_DIR.rglob(".txt"))
    pdfs = [p for p in pdfs if p.name not in ["app.py"]]
    for p in pdfs:
        st.markdown(f"<small>📄 {p.name}</small>", unsafe_allow_html=True)
    if not pdfs:
        st.markdown("<small>Ninguno</small>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Nueva conversacion", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["chain"] = None
        st.rerun()

st.markdown("""<div class="gcba-header"><div><h1>🏛️ Asistente de Consultas Internas</h1><p>Gobierno de la Ciudad Autonoma de Buenos Aires</p></div></div>""", unsafe_allow_html=True)

vs = build_vectorstore()
if vs:
    n = len(vs.docstore._dict)
    st.markdown(f'<span class="status-badge status-ok">Base activa · {n} fragmentos indexados</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-badge status-warn">Sin documentos PDF en el repositorio</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-label chat-label-user">Vos</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-label">Asistente GCBA</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("Ver fuentes"):
                for src in msg["sources"]:
                    st.markdown(f"- *{src['name']}* — {src['snippet']}")

if prompt := st.chat_input("Consulta normativas, organigrama, procedimientos..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    if not api_key:
        st.session_state["messages"].append({"role": "assistant", "content": "Necesito una Groq API Key. Configurala en el panel lateral.", "sources": []})
        st.rerun()
    if not vs:
        st.session_state["messages"].append({"role": "assistant", "content": "No hay documentos PDF en el repositorio.", "sources": []})
        st.rerun()
    if "chain" not in st.session_state or not st.session_state["chain"]:
        st.session_state["chain"] = build_chain(vs, api_key)
    chain = st.session_state["chain"]
    with st.spinner("Consultando..."):
        try:
            result = chain({"question": f"{SYSTEM_PROMPT}\n\nPregunta: {prompt}"})
            answer = result["answer"]
            sources = []
            seen = set()
            for doc in result.get("source_documents", []):
                name = doc.metadata.get("source_name", "Documento")
                if name not in seen:
                    seen.add(name)
                    sources.append({"name": name, "snippet": doc.page_content[:120].replace("\n", " ").strip() + "..."})
            st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {str(e)}", "sources": []})
    st.rerun()
