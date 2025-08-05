from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from tqdm import tqdm

DATA_DIR = "data"
INDEX_FILE = "vectorstore/myths.index"
DOCS_FILE = "vectorstore/myths.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Load documents from the data folder
def load_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                # Chunking the text by double newlines (naive but effective for myths)
                chunks = text.split("\n\n")
                documents.extend([chunk.strip() for chunk in chunks if chunk.strip()])
    return documents

# Create FAISS index
def build_faiss_index(docs):
    embeddings = model.encode(docs, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Save index and docs
def save(index, docs):
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

if __name__ == "__main__":
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} chunks")

    print("🔍 Building FAISS index...")
    index, embeddings = build_faiss_index(documents)

    print("💾 Saving index...")
    save(index, documents)
    print("✅ Done! Vector store is ready.")
