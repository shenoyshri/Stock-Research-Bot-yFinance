import streamlit as st
import pickle
import os
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Stock Research Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=DM+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.app-title { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; color: #0d1117; margin-bottom: 0; }
.app-subtitle { font-size: 1rem; color: #6b7280; margin-top: 4px; margin-bottom: 20px; }
.chat-user { background: #0d1117; color: #fff; border-radius: 12px 12px 4px 12px; padding: 12px 16px; margin: 8px 0; max-width: 80%; float: right; clear: both; font-size: 0.95rem; }
.chat-bot { background: #f6fff8; color: #0d1117; border-radius: 12px 12px 12px 4px; padding: 14px 18px; margin: 8px 0; max-width: 85%; float: left; clear: both; font-size: 0.95rem; border-left: 4px solid #22c55e; white-space: pre-wrap; }
.chat-container { overflow: auto; margin-bottom: 10px; }
.status-ok { display:inline-block; background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; border-radius:999px; padding:2px 12px; font-size:0.78rem; font-weight:600; }
.status-warn { display:inline-block; background:#fffbeb; color:#92400e; border:1px solid #fcd34d; border-radius:999px; padding:2px 12px; font-size:0.78rem; font-weight:600; }
.metric-box { background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.metric-label { font-size:0.78rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.metric-value { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:#0d1117; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="app-title">📈 Stock Research Bot</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">RAG chatbot powered by yFinance data · FAISS + Gemini 2.5 Flash</div>', unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")
    st.markdown("---")
    st.markdown("### 📂 FAISS Index & Documents")
    faiss_file = st.file_uploader("Upload `faiss_index.pkl`", type=["pkl"])
    docs_file  = st.file_uploader("Upload `documents.pkl`",   type=["pkl"])
    chunks_file = st.file_uploader("Upload `chunks.pkl`", type=["pkl"])
    st.markdown("---")
    st.markdown("""### ℹ️ About
Chat with a knowledge base built from **live yFinance data**.

**Knowledge base contains:**
- Company info (industry, sector, description)
- Latest 5 news headlines + summaries
- 30-day historical price data

**Pipeline:**
yFinance → SentenceTransformer → FAISS → Gemini answer
""")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages_yf = []
        st.session_state.chat_history_yf = []
        st.rerun()

# ─── Session State ─────────────────────────────────────────────────────────────
if "messages_yf"     not in st.session_state: st.session_state.messages_yf     = []
if "chat_history_yf" not in st.session_state: st.session_state.chat_history_yf = []
if "faiss_index"     not in st.session_state: st.session_state.faiss_index     = None
if "documents"       not in st.session_state: st.session_state.documents       = None
if "chunks"          not in st.session_state: st.session_state.chunks          = None
if "live_data"       not in st.session_state: st.session_state.live_data       = None

# ─── Load Pickles ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading index…")
def load_pkl_bytes(file_bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        obj = pickle.load(f)
    os.unlink(tmp_path)
    return obj

if faiss_file and docs_file:
    try:
        st.session_state.faiss_index = load_pkl_bytes(faiss_file.read())
        st.session_state.documents   = load_pkl_bytes(docs_file.read())
        st.session_state.chunks      = load_pkl_bytes(chunks_file.read())
        with col_status:
            st.markdown('<br><br><span class="status-ok">✅ Ready</span>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load files: {e}")
        with col_status:
            st.markdown('<br><br><span class="status-warn">❌ Error</span>', unsafe_allow_html=True)
elif faiss_file or docs_file:
    with col_status:
        st.markdown('<br><br><span class="status-warn">⚠️ Need Both</span>', unsafe_allow_html=True)
else:
    with col_status:
        st.markdown('<br><br><span class="status-warn">⚠️ Upload Files</span>', unsafe_allow_html=True)

# ─── Live Ticker Panel ─────────────────────────────────────────────────────────
st.markdown("---")
col_ticker, col_btn = st.columns([3, 1])
with col_ticker:
    ticker_sym = st.text_input("Ticker symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, INFY.NS")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("🔄 Fetch Live Data", use_container_width=True)

if refresh:
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker_sym)
        info  = stock.info
        hist  = stock.history(period="5d")
        st.session_state.live_data = {"info": info, "hist": hist, "ticker": ticker_sym}
    except Exception as e:
        st.warning(f"Could not fetch live data: {e}")

if st.session_state.live_data:
    ld   = st.session_state.live_data
    info = ld["info"]
    hist = ld["hist"]
    c1, c2, c3, c4 = st.columns(4)
    price = hist["Close"].iloc[-1] if not hist.empty else None
    prev  = hist["Close"].iloc[-2] if len(hist) > 1 else None
    delta_str = f" ({((price - prev)/prev*100):+.2f}%)" if (price and prev) else ""
    price_str = f"${price:.2f}{delta_str}" if price else "N/A"
    mktcap    = info.get("marketCap", None)
    mktcap_str = f"${mktcap/1e9:.1f}B" if isinstance(mktcap, (int, float)) else "N/A"
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Last Price</div><div class="metric-value">{price_str}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Market Cap</div><div class="metric-value">{mktcap_str}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Sector</div><div class="metric-value">{info.get("sector","N/A")}</div></div>', unsafe_allow_html=True)
    with c4:
        ind = info.get("industry", "N/A")
        st.markdown(f'<div class="metric-box"><div class="metric-label">Industry</div><div class="metric-value">{ind[:24]}</div></div>', unsafe_allow_html=True)
    with st.expander("📊 5-Day Closing Price Chart"):
        if not hist.empty:
            st.line_chart(hist["Close"])

# ─── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_context(query, top_k=3):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_emb = model.encode([query])
    distances, indices = st.session_state.faiss_index.search(
        np.array(query_emb).astype("float32"), k=top_k
    )
    chunks = st.session_state.chunks
    return "\n\n".join([chunks[i].page_content for i in indices[0] if i < len(chunks)])

def ask_question_yf(question, api_key):
    from google import genai
    client  = genai.Client(api_key=api_key)
    context = retrieve_context(question)
    history = "\n".join(st.session_state.chat_history_yf)
    prompt  = f"""You are an AI Financial Assistant specialized in stock market analysis.
Use the provided context to answer the user's question accurately.

Conversation History:
{history}

Context:
{context}

User Question: {question}

Instructions:
- Answer only using the provided context
- If not available say: 'I could not find enough information in the available financial data.'
- Summarize financial news clearly
- Explain stock trends in simple language
- Be concise but informative
- Do not fabricate financial information
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

# ─── Chat Display ──────────────────────────────────────────────────────────────
st.markdown("---")
if not st.session_state.messages_yf:
    st.info("💡 Try: *'What is the stock price of Apple?'*  ·  *'Describe the industry and sector.'*  ·  *'What are the recent news headlines?'*")

for msg in st.session_state.messages_yf:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-container"><div class="chat-user">🧑 {msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container"><div class="chat-bot">🤖 {msg["content"]}</div></div>', unsafe_allow_html=True)

# ─── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.form("qform_yf", clear_on_submit=True):
    col_i, col_b = st.columns([5, 1])
    with col_i:
        user_input = st.text_input("Question", label_visibility="collapsed", placeholder="Ask about the stock data…")
    with col_b:
        submitted = st.form_submit_button("Ask →", use_container_width=True)

if submitted and user_input:
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar.")
    elif st.session_state.faiss_index is None or st.session_state.chunks is None:
        st.warning("Upload both `faiss_index.pkl` and `documents.pkl` in the sidebar.")
    else:
        st.session_state.messages_yf.append({"role": "user", "content": user_input})
        with st.spinner("Searching knowledge base and generating answer…"):
            try:
                answer = ask_question_yf(user_input, api_key)
                st.session_state.messages_yf.append({"role": "assistant", "content": answer})
                st.session_state.chat_history_yf += [f"Question: {user_input}", f"Answer: {answer}"]
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
