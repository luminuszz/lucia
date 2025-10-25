#!/home/daviribeiro/projects/personal/lucia/venv/bin/python

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import torch

# === CONFIGURAÇÕES ===
TRANSCRIPTIONS_DIR = "/home/daviribeiro/projects/personal/lucia/transcricoes"
LLM_MODEL_PATH = "/home/daviribeiro/projects/personal/lucia/models/guanaco-7b-uncensored.Q4_K_M.gguf"
TOP_K = 5
MAX_CONTEXT_TOKENS = 1500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === INFORMAÇÕES DA GPU ===
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🔍 GPU detectada: {gpu_name}")
    print(f"    Total de memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ GPU não disponível, usando CPU.")

# === CARREGANDO MODELO DE EMBEDDINGS ===
print(f"🔍 Carregando modelo de embeddings ({DEVICE})...")
embed_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

# === CARREGANDO LLM LOCAL GGUF ===
print("🔍 Carregando LLM local GGUF na GPU...")
llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048, n_threads=8, verbose=False, gpu_layers=40)

# === CARREGANDO E CHUNKING DAS TRANSCRIÇÕES ===
print("📄 Processando transcrições...")
file_names = sorted([f.name for f in Path(TRANSCRIPTIONS_DIR).glob("*.md")])
docs = []
file_index = []

for fname in file_names:
    with open(os.path.join(TRANSCRIPTIONS_DIR, fname), "r", encoding="utf-8") as f:
        text = f.read()
        # Chunking inteligente (aprox. 200 tokens por chunk)
        approx_chunk_size = 800  # caracteres (~200 tokens)
        chunks = [text[i:i+approx_chunk_size] for i in range(0, len(text), approx_chunk_size)]
        docs.extend(chunks)
        file_index.extend([fname]*len(chunks))

# Encode todos os chunks na GPU
embeddings = embed_model.encode(docs, convert_to_tensor=True, device=DEVICE)
print(f"✅ {len(file_names)} transcrições carregadas, {len(docs)} chunks gerados.\n")

# === LISTAR ARQUIVOS ===
for i, fname in enumerate(file_names, 1):
    print(f"{i} - {fname}")
print("0 - Buscar em todas as transcrições")

# === FUNÇÃO RETRIEVER ===
def retrieve_chunks(query, top_k=TOP_K, selected_files=None):
    with torch.no_grad():
        query_emb = embed_model.encode(query, convert_to_tensor=True, device=DEVICE)
        scores = util.cos_sim(query_emb, embeddings).squeeze(0)

    scores_cpu = scores.cpu()
    
    if selected_files:
        filtered_indices = [i for i, f in enumerate(file_index) if f in selected_files]
        filtered_scores = scores_cpu[filtered_indices]
        top_indices = torch.topk(filtered_scores, k=min(top_k, len(filtered_scores))).indices
        top_indices = [filtered_indices[i] for i in top_indices]
    else:
        top_indices = torch.topk(scores_cpu, k=min(top_k, len(docs))).indices.tolist()

    # Seleciona chunks respeitando MAX_CONTEXT_TOKENS
    selected_chunks = []
    tokens_so_far = 0
    for i in top_indices:
        chunk = docs[i]
        approx_tokens = len(chunk) // 4
        if tokens_so_far + approx_tokens > MAX_CONTEXT_TOKENS:
            break
        selected_chunks.append({"text": chunk, "file": file_index[i]})
        tokens_so_far += approx_tokens

    return selected_chunks

# === FUNÇÃO READER ===
def ask_llm(chunks, query, mode="trecho"):
    """
    mode:
        trecho -> retorna o texto relacionado
        resumo -> resume o contexto
        explicacao -> explica o conteúdo
    """
    chunk_texts = "\n\n".join([f"[{c['file']}]: {c['text']}" for c in chunks])
    
    if mode == "trecho":
        prompt = f"Você é um assistente. Retorne os trechos relevantes para a pergunta.\n\n{chunk_texts}\n\nPergunta: {query}\nResposta:"
    elif mode == "resumo":
        prompt = f"Você é um assistente. Resuma os trechos abaixo de forma clara:\n\n{chunk_texts}\n\nResumo:"
    elif mode == "explicacao":
        prompt = f"Você é um assistente. Explique os trechos abaixo de forma detalhada, facilitando o entendimento:\n\n{chunk_texts}\n\nExplicação:"
    else:
        prompt = f"{chunk_texts}\nPergunta: {query}\nResposta:"

    try:
        resp = llm(prompt=prompt, max_tokens=300, temperature=0.2)
        return resp["choices"][0]["text"].strip()
    except ValueError:
        return "❌ Erro: o prompt excedeu a capacidade do modelo. Tente perguntas menores ou menos contexto."

# === LOOP PRINCIPAL ===
choice = int(input("\nEscolha uma transcrição pelo número ou 0 para todas: "))
selected_files = None if choice == 0 else [file_names[choice-1]]

while True:
    query = input("\nPergunta (digite 'sair' para encerrar): ")
    if query.lower() == "sair":
        break

    mode = input("Modo (trecho/resumo/explicacao): ").lower()
    if mode not in ["trecho", "resumo", "explicacao"]:
        mode = "trecho"

    chunks = retrieve_chunks(query, selected_files=selected_files)
    if not chunks:
        print("❌ Nenhum conteúdo encontrado para essa pergunta.")
        continue

    answer = ask_llm(chunks, query, mode=mode)
    print(f"\n💬 {mode.capitalize()}:\n{answer}")
