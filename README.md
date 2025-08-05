# 🧠 Medical Myths Bot (RAG + LLaMA 3)

A **Retrieval-Augmented Generation (RAG)** chatbot that debunks common **medical myths** using a vector database and **LLaMA 3** for high-quality natural language answers.

Built with:
- 🔍 **FAISS** vector search
- 🔢 **E5-small** for embeddings
- 🦙 **Meta LLaMA 3-8B-Instruct** via Hugging Face Inference API
- 💬 **Python app** (CLI-based)

## 💡 Example

❓ Does using a microwave cause cancer?

🤖 Answer: There is no conclusive evidence to suggest that using a microwave oven increases your risk of developing cancer...

---

## 📦 Project Structure

- `data/mythbusters.txt` – Raw myth-busting text
- `vectorstore/` – FAISS index and stored chunks
- `ingest.py` – Embeds text into FAISS
- `app.py` – Chat loop powered by vector search + LLaMA 3

---

## 🛠️ Setup

### Prerequisites
- **Python 3.10** (or later)
- **Git LFS** for handling large files

### Steps to Run

1. **Clone the repo**:
```bash
git clone 
cd medical-myths-bot
```
2. Create .env file and add your Hugging Face token:
```
HF_TOKEN=your_huggingface_token_here
```
3. Install dependencies:
```
pip install -r requirements.txt
```
5. Ingest your data into the FAISS index:
```
python ingest.py
```
6. Run the bot:
```
python app.py
```

🧠 Built By
Swini Rodrigues | GitHub

