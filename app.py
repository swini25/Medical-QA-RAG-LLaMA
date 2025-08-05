import os
import pickle
import faiss
import torch
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModel

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Constants
INDEX_FILE = "vectorstore/myths.index"
DOCS_FILE = "vectorstore/myths.pkl"
EMBED_MODEL = "intfloat/e5-small"

# Load FAISS index and documents
print("📂 Loading vector store and documents...")
index = faiss.read_index(INDEX_FILE)
with open(DOCS_FILE, "rb") as f:
    documents = pickle.load(f)

# Load embedding model
print("🔢 Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_query(query):
    inputs = embed_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = embed_model(**inputs)
    return mean_pooling(model_output, inputs["attention_mask"]).squeeze().numpy()

def retrieve_relevant_chunks(query_vector, k=3):
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    return [documents[i] for i in indices[0]]

# Load LLaMA 3 Inference API
print("🧠 Connecting to LLaMA 3 on Hugging Face...")
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token
)

def ask_llama3(question, context):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful medical assistant that debunks health and science myths using reliable evidence and clear language.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Using the context below, answer the question in 2–3 factual, conversational sentences.

Context:
{context}

Question:
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    response = client.text_generation(
        prompt,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.1,
        stop_sequences=["<|eot_id|>"]
    )
    return response.strip()

# Main loop
if __name__ == "__main__":
    while True:
        user_input = input("\n❓ Ask a medical myth (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        query_vector = embed_query(user_input)
        top_chunks = retrieve_relevant_chunks(query_vector)
        combined_context = "\n".join(top_chunks)

        answer = ask_llama3(user_input, combined_context)

        print("\n🤖 Answer:")
        print(answer)
