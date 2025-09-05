import os
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------
# CONFIGURAÇÕES
# --------------------------
TRANSCRIPTS_DIR = "transcricoes"  # pasta com arquivos .md ou .txt
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TheBloke/guanaco-7B-GGML"  # ou outro LLM local quantizado
VECTOR_DIM = 384  # dimensão do embedding (MiniLM-L6-v2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# FUNÇÃO PARA CARREGAR TRANSCRIÇÕES
# --------------------------
def load_transcripts(directory):
    transcripts = []
    files = []
    for file in Path(directory).glob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            transcripts.append(text)
            files.append(file.name)
    return transcripts, files

# --------------------------
# CRIA OU CARREGA INDICE FAISS
# --------------------------
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(embeddings)
    return index

# --------------------------
# INICIALIZA MODELOS
# --------------------------
print("🔍 Carregando modelo de embeddings...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

print("🔍 Carregando modelo local LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer, device=0 if DEVICE=="cuda" else -1)

# --------------------------
# CARREGA TRANSCRIÇÕES E CRIA EMBEDDINGS
# --------------------------
transcripts, files = load_transcripts(TRANSCRIPTS_DIR)
print(f"📄 {len(transcripts)} transcrições carregadas.")

print("🟢 Criando embeddings...")
embeddings = embed_model.encode(transcripts, convert_to_numpy=True)

# --------------------------
# CRIA INDICE FAISS
# --------------------------
print("🟢 Criando índice FAISS...")
index = create_faiss_index(embeddings)

# --------------------------
# LOOP DE PERGUNTAS
# --------------------------
print("▶️ Pronto para perguntas sobre as transcrições! Digite 'sair' para encerrar.")

while True:
    query = input("\nDigite sua pergunta: ")
    if query.lower() in ["sair", "exit"]:
        break

    # gerar embedding da query
    query_vec = embed_model.encode([query], convert_to_numpy=True)

    # buscar nos embeddings
    D, I = index.search(query_vec, k=3)  # top 3 resultados
    context_texts = "\n\n".join([transcripts[i] for i in I[0]])

    # criar prompt para LLM local
    prompt = f"Baseado nos seguintes trechos de transcrição:\n{context_texts}\n\nResponda à pergunta: {query}"

    # gerar resposta
    response = llm_pipeline(prompt, max_length=512, do_sample=True, temperature=0.7)[0]["generated_text"]

    print("\n💡 Resposta:")
    print(response)
