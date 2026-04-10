import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import Document
import re

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
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
.confidence-high { background: #d4edda; color: #155724; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.7rem; }
.confidence-med  { background: #fff3cd; color: #856404;  padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.7rem; }
.confidence-low  { background: #f8d7da; color: #721c24;  padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.7rem; }
.no-context-msg  { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; padding: 0.75rem 1rem; border-radius: 4px; margin: 0.5rem 0; }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES Y CONFIGURACIÓN
# ─────────────────────────────────────────────
DOCS_DIR = Path(".")

# MEJORA 1: Modelo de embeddings más potente para español
# paraphrase-multilingual-mpnet-base-v2 tiene mejor rendimiento en español
# que MiniLM, especialmente con texto formal/burocrático argentino.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Umbral mínimo de similitud coseno para considerar un chunk relevante.
# Por debajo de este valor, se considera que no hay contexto suficiente.
RELEVANCE_THRESHOLD = 0.35

# ─────────────────────────────────────────────
# PROMPTS — Anti-alucinación
# ─────────────────────────────────────────────

# MEJORA 2: System prompt separado (no pegado en la pregunta) con instrucciones
# explícitas de anti-alucinación. Las instrucciones clave son:
#   - NUNCA inventar información no presente en los documentos
#   - Indicar claramente cuando la información no está disponible
#   - No extrapolar ni completar con conocimiento general
SYSTEM_TEMPLATE = """Sos el asistente oficial del Gobierno de la Ciudad Autónoma de Buenos Aires (GCBA).
Tu función es responder consultas de empleados sobre normativas, procedimientos y organigrama,
basándote EXCLUSIVAMENTE en los documentos internos que se te proporcionan como contexto.

REGLAS ESTRICTAS ANTI-ALUCINACIÓN:
1. Solo respondé con información que esté explícitamente en el contexto proporcionado.
2. Si la información no está en el contexto, decí exactamente: "Esta información no se encuentra en los documentos disponibles."
3. NUNCA inventes datos, nombres, fechas, números de resolución o procedimientos.
4. NUNCA uses tu conocimiento general para completar información que no está en el contexto.
5. Si el contexto es parcial, indicá qué parte pudiste responder y qué no.
6. Citá siempre la fuente (nombre del documento) de donde tomás la información.
7. Si hay contradicciones entre documentos, mencionálas explícitamente.

FORMATO DE RESPUESTA:
- Usá español rioplatense, claro y profesional.
- Sé conciso pero completo.
- Estructurá la respuesta con bullets cuando sea apropiado.
- Incluí siempre la fuente al final: "Fuente: [nombre del documento]"

Contexto de documentos:
──────────────────────
{context}
──────────────────────

Historial de conversación:
{chat_history}"""

HUMAN_TEMPLATE = "Pregunta: {question}"

# ─────────────────────────────────────────────
# FUNCIONES DE EMBEDDINGS Y VECTORSTORE
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Indexando documentos...")
def build_vectorstore():
    embeddings = get_embeddings()
    all_docs = []

    # MEJORA 3: Chunks más grandes con más overlap para mejor contexto.
    # chunk_size=1000 (era 800) captura más contexto por fragmento.
    # chunk_overlap=200 (era 120) reduce el riesgo de cortar ideas a la mitad.
    # separators priorizan cortes en párrafos > oraciones > palabras.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    for filepath in DOCS_DIR.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(filepath))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for i, c in enumerate(chunks):
                c.metadata["source_name"] = filepath.name
                c.metadata["chunk_index"] = i
                c.metadata["total_chunks"] = len(chunks)
            all_docs.extend(chunks)
        except Exception as e:
            st.warning(f"No se pudo cargar {filepath.name}: {e}")

    for filepath in DOCS_DIR.rglob("*.txt"):
        try:
            loader = TextLoader(str(filepath), encoding="utf-8")
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for i, c in enumerate(chunks):
                c.metadata["source_name"] = filepath.name
                c.metadata["chunk_index"] = i
                c.metadata["total_chunks"] = len(chunks)
            all_docs.extend(chunks)
        except Exception as e:
            st.warning(f"No se pudo cargar {filepath.name}: {e}")

    if not all_docs:
        return None

    return FAISS.from_documents(all_docs, embeddings)


def get_relevant_docs_with_scores(vectorstore, query: str, k: int = 6) -> list[tuple[Document, float]]:
    """
    MEJORA 4: Retrieval con score de relevancia + filtro por umbral.
    
    Usa similarity_search_with_score para obtener la distancia coseno de cada
    chunk. Los chunks con score < RELEVANCE_THRESHOLD se descartan, evitando
    que el LLM reciba contexto irrelevante que podría inducir alucinaciones.
    
    FAISS devuelve distancia L2 (menor = más similar). La convertimos a
    similitud coseno: sim = 1 - (dist / 2) para embeddings normalizados.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    filtered = []
    for doc, dist in results:
        # Convertir distancia L2 a similitud coseno (embeddings normalizados)
        similarity = max(0.0, 1.0 - (dist / 2.0))
        if similarity >= RELEVANCE_THRESHOLD:
            doc.metadata["relevance_score"] = round(similarity, 3)
            filtered.append((doc, similarity))
    
    # Ordenar de mayor a menor similitud
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered


def assess_context_quality(docs_with_scores: list) -> dict:
    """
    MEJORA 5: Evaluación de la calidad del contexto recuperado.
    
    Calcula métricas para informar al usuario sobre la confiabilidad de la respuesta:
    - has_context: si hay al menos un chunk relevante
    - avg_score: promedio de similitud del contexto
    - confidence: nivel de confianza basado en scores
    - unique_sources: cantidad de documentos únicos usados
    """
    if not docs_with_scores:
        return {
            "has_context": False,
            "avg_score": 0.0,
            "confidence": "sin_contexto",
            "unique_sources": 0
        }
    
    scores = [s for _, s in docs_with_scores]
    avg = sum(scores) / len(scores)
    sources = set(doc.metadata.get("source_name", "") for doc, _ in docs_with_scores)
    
    if avg >= 0.65:
        confidence = "alta"
    elif avg >= 0.50:
        confidence = "media"
    else:
        confidence = "baja"
    
    return {
        "has_context": True,
        "avg_score": round(avg, 3),
        "confidence": confidence,
        "unique_sources": len(sources)
    }


def build_chain(vectorstore, api_key: str):
    """
    MEJORA 6: Cadena con prompt estructurado real (SystemMessage + HumanMessage)
    en lugar de pegar el system prompt dentro de la pregunta del usuario.
    
    Esto hace que el LLM trate las instrucciones como system context y la
    pregunta como input del usuario, respetando el rol de cada parte.
    """
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,       # Más bajo (era 0.2) → menos creatividad → menos alucinación
        max_tokens=1500,       # Más tokens para respuestas completas
        model_kwargs={
            "top_p": 0.9,      # Nucleus sampling: limita tokens improbables
        }
    )

    # MEJORA 7: Memoria con límite de tokens, no solo de mensajes.
    # k=5 (era 6) con return_messages=True para chat format correcto.
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # MEJORA 8: Prompt estructurado con instrucciones anti-alucinación
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context", "chat_history"],
            template=SYSTEM_TEMPLATE
        )
    )
    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template=HUMAN_TEMPLATE
        )
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # MEJORA 9: MMR (Maximal Marginal Relevance) para diversidad en el retrieval.
    # En lugar de traer los k chunks más similares (que pueden ser repetitivos),
    # MMR balancea similitud con diversidad, cubriendo más aspectos del tema.
    # fetch_k=12: busca 12 candidatos, selecciona los 6 más diversos y relevantes.
    # lambda_mult=0.7: 70% peso a relevancia, 30% a diversidad.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 12,
            "lambda_mult": 0.7
        }
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
        verbose=False
    )


def detect_no_context_response(answer: str) -> bool:
    """
    MEJORA 10: Detección de respuestas donde el LLM indicó falta de información.
    
    Verifica si la respuesta contiene frases que indican que el modelo
    no encontró información en los documentos (siguiendo las instrucciones
    del system prompt).
    """
    no_context_phrases = [
        "no se encuentra en los documentos",
        "no está en los documentos",
        "no tengo información",
        "no hay información disponible",
        "esta información no",
        "los documentos disponibles no",
        "no puedo encontrar",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in no_context_phrases)


def format_confidence_badge(confidence: str) -> str:
    if confidence == "alta":
        return '<span class="confidence-high">● Confianza alta</span>'
    elif confidence == "media":
        return '<span class="confidence-med">● Confianza media</span>'
    elif confidence == "baja":
        return '<span class="confidence-low">● Confianza baja</span>'
    else:
        return '<span class="confidence-low">● Sin contexto</span>'


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Asistente GCBA")
    st.markdown("---")
    st.markdown("### Configuracion")
    api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input(
        "Groq API Key", type="password", placeholder="gsk_..."
    )

    st.markdown("---")
    st.markdown("### Umbral de relevancia")
    threshold = st.slider(
        "Similitud mínima", 0.20, 0.70, RELEVANCE_THRESHOLD, 0.05,
        help="Chunks con similitud menor a este valor se ignoran. "
             "Valores altos = más preciso pero puede no encontrar contexto. "
             "Valores bajos = más contexto pero más riesgo de ruido."
    )
    RELEVANCE_THRESHOLD = threshold

    st.markdown("---")
    st.markdown("### Documentos cargados")
    pdfs = list(DOCS_DIR.rglob("*.pdf")) + list(DOCS_DIR.rglob("*.txt"))
    pdfs = [p for p in pdfs if p.name not in ["app.py", "app_mejorado.py"]]
    for p in pdfs:
        st.markdown(f"<small>📄 {p.name}</small>", unsafe_allow_html=True)
    if not pdfs:
        st.markdown("<small>Ninguno</small>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Nueva conversacion", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["chain"] = None
        st.rerun()

    # Info de la sesión
    if st.session_state.get("messages"):
        n_msgs = len(st.session_state["messages"])
        st.markdown(f"<small>💬 {n_msgs} mensajes en esta sesión</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="gcba-header">
  <div>
    <h1>🏛️ Asistente de Consultas Internas</h1>
    <p>Gobierno de la Ciudad Autonoma de Buenos Aires · Solo responde con informacion de documentos oficiales</p>
  </div>
</div>
""", unsafe_allow_html=True)

vs = build_vectorstore()
if vs:
    n = len(vs.docstore._dict)
    st.markdown(
        f'<span class="status-badge status-ok">Base activa · {n} fragmentos indexados · '
        f'Modelo: {EMBED_MODEL.split("/")[-1]}</span>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="status-badge status-warn">Sin documentos PDF en el repositorio</span>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HISTORIAL DE CHAT
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown('<div class="chat-label chat-label-user">Vos</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-label">Asistente GCBA</div>', unsafe_allow_html=True)

        # Badge de confianza
        confidence = msg.get("confidence", "")
        if confidence:
            st.markdown(format_confidence_badge(confidence), unsafe_allow_html=True)

        # Estilo diferente si no hay contexto
        if msg.get("no_context"):
            st.markdown(
                f'<div class="no-context-msg">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message-bot">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        # Fuentes con score de relevancia
        if msg.get("sources"):
            with st.expander(f"Ver fuentes ({len(msg['sources'])} fragmentos)"):
                for src in msg["sources"]:
                    score_pct = int(src.get("score", 0) * 100)
                    score_color = "#155724" if score_pct >= 65 else "#856404" if score_pct >= 50 else "#721c24"
                    st.markdown(
                        f"- **{src['name']}** "
                        f"<span style='color:{score_color};font-size:0.75rem'>"
                        f"[relevancia: {score_pct}%]</span><br>"
                        f"  <small><em>{src['snippet']}</em></small>",
                        unsafe_allow_html=True
                    )

# ─────────────────────────────────────────────
# INPUT Y LÓGICA DE RESPUESTA
# ─────────────────────────────────────────────
if prompt := st.chat_input("Consulta normativas, organigrama, procedimientos..."):
    # Sanitizar input
    prompt = prompt.strip()
    if len(prompt) > 1000:
        prompt = prompt[:1000]

    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Validaciones previas
    if not api_key:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "⚠️ Necesito una Groq API Key. Configurala en el panel lateral.",
            "sources": [],
            "confidence": "sin_contexto",
            "no_context": True
        })
        st.rerun()

    if not vs:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "⚠️ No hay documentos PDF en el repositorio. Agregá documentos para poder responder consultas.",
            "sources": [],
            "confidence": "sin_contexto",
            "no_context": True
        })
        st.rerun()

    # MEJORA 11: Pre-verificación de relevancia antes de llamar al LLM.
    # Si ningún chunk supera el umbral, se rechaza la consulta directamente
    # sin gastar tokens ni arriesgar una alucinación.
    docs_with_scores = get_relevant_docs_with_scores(vs, prompt)
    context_quality = assess_context_quality(docs_with_scores)

    if not context_quality["has_context"]:
        no_context_msg = (
            "No encontré información relacionada con tu consulta en los documentos disponibles. "
            "Te sugiero reformular la pregunta o verificar si el tema está cubierto en la documentación cargada. "
            "Solo puedo responder sobre el contenido de los documentos internos del GCBA."
        )
        st.session_state["messages"].append({
            "role": "assistant",
            "content": no_context_msg,
            "sources": [],
            "confidence": "sin_contexto",
            "no_context": True
        })
        st.rerun()

    # Construir/reutilizar la cadena
    if "chain" not in st.session_state or not st.session_state["chain"]:
        st.session_state["chain"] = build_chain(vs, api_key)

    chain = st.session_state["chain"]

    with st.spinner("Consultando documentos..."):
        try:
            result = chain({"question": prompt})
            answer = result["answer"]

            # Detectar si el LLM indicó falta de información
            no_context_detected = detect_no_context_response(answer)

            # Procesar fuentes con scores
            sources = []
            seen = set()
            for doc in result.get("source_documents", []):
                name = doc.metadata.get("source_name", "Documento")
                score = doc.metadata.get("relevance_score", 0)
                if name not in seen:
                    seen.add(name)
                    sources.append({
                        "name": name,
                        "snippet": doc.page_content[:150].replace("\n", " ").strip() + "...",
                        "score": score
                    })

            # Ordenar fuentes por score
            sources.sort(key=lambda x: x["score"], reverse=True)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": "sin_contexto" if no_context_detected else context_quality["confidence"],
                "no_context": no_context_detected
            })

        except Exception as e:
            error_msg = str(e)
            # No exponer detalles técnicos al usuario
            if "rate_limit" in error_msg.lower():
                user_msg = "⚠️ Límite de la API alcanzado. Esperá unos segundos y volvé a intentar."
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                user_msg = "⚠️ API Key inválida. Verificá la clave en el panel lateral."
            else:
                user_msg = f"⚠️ Ocurrió un error al procesar tu consulta. Intentá de nuevo."

            st.session_state["messages"].append({
                "role": "assistant",
                "content": user_msg,
                "sources": [],
                "confidence": "sin_contexto",
                "no_context": True
            })

    st.rerun()
