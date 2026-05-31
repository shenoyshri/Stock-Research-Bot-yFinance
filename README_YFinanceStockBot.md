# 📈 Stock Research Bot (yFinance)

A RAG-powered stock market chatbot that answers questions from a knowledge base built with live yFinance data — company fundamentals, recent news headlines, and 30-day historical prices. Includes a real-time ticker dashboard with price, market cap, sector, and a 5-day chart.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install streamlit langchain langchain-community langchain-text-splitters \
            sentence-transformers faiss-cpu yfinance numpy google-genai

# 2. Run the app
streamlit run app2_yfinance_stock_bot.py
```

---

## 📂 Required Files

| File | Description |
|------|-------------|
| `app2_yfinance_stock_bot.py` | The Streamlit app |
| `faiss_index.pkl` | Pre-built FAISS flat L2 index (upload via sidebar) |
| `documents.pkl` | Raw document strings used to build the index (upload via sidebar) |

Place all three files in the same folder before running.

---

## ⚙️ Configuration (Sidebar)

| Setting | Details |
|---------|---------|
| **Google Gemini API Key** | Get from [aistudio.google.com](https://aistudio.google.com/app/apikey) — entered at runtime, never stored |
| **faiss_index.pkl** | Upload via sidebar — the FAISS vector index |
| **documents.pkl** | Upload via sidebar — the raw document chunks used to build the index |

Both pickle files must be uploaded together. The app re-chunks `documents.pkl` at load time to reconstruct the text lookup table that maps FAISS indices back to readable content.

---

## 🏗️ Architecture

```
yFinance API  (Ticker: e.g. AAPL)
       ↓
Three knowledge sources assembled:
  1. Company Info  (industry, sector, business description)
  2. Latest 5 News (title + summary per article)
  3. 30-day Historical Price Data (OHLCV table)
       ↓
RecursiveCharacterTextSplitter
  chunk_size=100 · overlap=0
       ↓
SentenceTransformer
  model: all-MiniLM-L6-v2
       ↓
FAISS IndexFlatL2  ──→  faiss_index.pkl + documents.pkl
       ↓
Semantic Retrieval  (top-3 chunks)
       ↓
Gemini 2.5 Flash
  (answer + multi-turn chat history)
```

---

## 📊 Live Ticker Dashboard

Enter any valid ticker symbol (e.g. `AAPL`, `TSLA`, `NVDA`, `INFY.NS`) and click **Fetch Live Data** to get:

| Metric | Details |
|--------|---------|
| Last Price | Latest closing price with day-over-day % change |
| Market Cap | In billions USD |
| Sector | e.g. Technology, Healthcare |
| Industry | e.g. Consumer Electronics |
| 5-Day Chart | Interactive line chart of closing prices |

> **Note:** The live dashboard fetches fresh data from yFinance at click time. The Q&A chatbot answers from the pre-built pickle knowledge base (AAPL by default), not from the live fetch.

---

## 💬 Sample Questions

| Category | Question |
|----------|---------|
| Price lookup | *"What is the stock price of Apple?"* |
| Company profile | *"What industry and sector does the company operate in?"* |
| Business summary | *"Describe what this company does."* |
| News | *"What are the recent news headlines?"* |
| Price history | *"How has the stock performed over the last 30 days?"* |
| Multi-turn | *"What was the highest closing price last month?"* |

---

## 🔄 Real-World Usage Scenarios

**Day traders** — Get a quick briefing on a stock's recent news and price trend before placing a trade.

**Equity researchers** — Rapidly profile a company's sector, industry, and recent sentiment without switching between tools.

**Finance students** — Learn to ask structured questions about real stock data and understand how RAG connects raw financial data to natural language answers.

**Portfolio monitoring** — Re-run the data pipeline for different tickers (TSLA, NVDA, RELIANCE.NS) and swap the pickle files to query a different stock's knowledge base.

**Interview prep** — Practice explaining stock performance and company fundamentals by interacting with a grounded AI that cites actual data.

---

## 🔁 Rebuilding the Knowledge Base for a Different Stock

The included pickle files are built on AAPL data. To create a knowledge base for a different ticker:

```python
import yfinance as yf, pickle, numpy as np, faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ticker = "TSLA"   # change this
stock  = yf.Ticker(ticker)
info   = stock.info
news   = stock.news

documents = []
documents.append(f"Industry: {info['industry']}, Sector: {info['sector']}, Description: {info['longBusinessSummary']}")
news_text = "\n".join([f"{n['content']['title']}\n{n['content']['summary']}" for n in news[:5]])
documents.append(news_text)
documents.append(stock.history(period="1mo").tail(30).to_string())

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks   = splitter.create_documents(documents)
model    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs     = model.encode([c.page_content for c in chunks])

index = faiss.IndexFlatL2(embs.shape[1])
index.add(np.array(embs).astype("float32"))

with open("faiss_index.pkl", "wb") as f: pickle.dump(index, f)
with open("documents.pkl",   "wb") as f: pickle.dump(documents, f)
print("Done — upload the new pkl files in the app sidebar.")
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Stock Data | yFinance |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | FAISS IndexFlatL2 (CPU) |
| LLM | Google Gemini 2.5 Flash |
| Serialization | Python pickle |

---

## ⚠️ Notes

- `faiss_index.pkl` and `documents.pkl` must always be uploaded **together** — the index stores vectors only; the documents file holds the text needed to reconstruct readable context.
- Both pickles were built with `all-MiniLM-L6-v2`. The retriever uses the same model at query time, so embedding dimensions always match.
- yFinance data is fetched live only for the dashboard panel. The chatbot Q&A always uses the uploaded pickle knowledge base.

---

## 👨‍💻 Author

**Srikanth Shenoy** · [GitHub](https://github.com/shenoyshri) · [LinkedIn](https://linkedin.com/in/shenoysrikanthp) · [Portfolio](https://shenoyshri.github.io)
