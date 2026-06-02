#!/home/daviribeiro/projects/personal/lucia/.venv/bin/python

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import torch
from llama_cpp import Llama
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# -------------------------------
# Configurações Globais
# -------------------------------

TRANSCRIPTIONS_DIR = (
    "/home/daviribeiro/Documents/obsidian-storage/davi-brain/transcricoes"
)
LLM_MODEL_PATH = (
    "/home/daviribeiro/projects/personal/lucia/models/guanaco-7b-uncensored.Q4_K_M.gguf"
)
CACHE_PATH = "/home/daviribeiro/projects/personal/lucia/.embeddings_cache.pt"
CACHE_META_PATH = "/home/daviribeiro/projects/personal/lucia/.embeddings_meta.json"

EMBED_MODEL_NAME = "all-mpnet-base-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVER = 10  # quantos chunks o retriever busca antes do re-ranking
TOP_K_FINAL = 5  # quantos chunks chegam ao LLM após re-ranking
MAX_CONTEXT_TOKENS = 1500

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150  # sobreposição entre chunks

# Memória reservada para o sistema (em bytes) — evita OOM
VRAM_SAFETY_MARGIN_GB = 1.0
# Tamanho estimado por camada do modelo em VRAM (7B Q4 ≈ 90 MB/camada)
VRAM_PER_LAYER_GB = 0.090
TOTAL_LAYERS = 32

# -------------------------------
# Logging
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("edmundo")

# -------------------------------
# GPU
# -------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / 1e9
    log.info("GPU detectada: %s (%.2f GB VRAM)", props.name, total_vram_gb)
else:
    total_vram_gb = 0.0
    log.warning("GPU não disponível, usando CPU.")


# -------------------------------
# MELHORIA 3 — gpu_layers calculado
# -------------------------------


def compute_gpu_layers() -> int:
    """
    Estima quantas camadas cabem na VRAM disponível,
    deixando VRAM_SAFETY_MARGIN_GB de folga para o sistema.
    """
    if DEVICE != "cuda":
        return 0
    usable_gb = max(0.0, total_vram_gb - VRAM_SAFETY_MARGIN_GB)
    layers = int(usable_gb / VRAM_PER_LAYER_GB)
    layers = min(layers, TOTAL_LAYERS)
    log.info(
        "VRAM utilizável: %.2f GB → %d camadas na GPU (de %d totais)",
        usable_gb,
        layers,
        TOTAL_LAYERS,
    )
    return layers


# -------------------------------
# Carregamento de modelos
# -------------------------------

log.info("Carregando modelo de embeddings '%s'...", EMBED_MODEL_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

log.info("Carregando Cross-Encoder para re-ranking...")
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

gpu_layers = compute_gpu_layers()
log.info("Carregando LLM GGUF com %d camadas na GPU...", gpu_layers)
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    verbose=False,
    n_gpu_layers=gpu_layers,  # parâmetro correto para llama-cpp-python >= 0.2
)
log.info("LLM carregado.")


# -------------------------------
# MELHORIA 2 — Chunking com sobreposição
# -------------------------------


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Divide texto em fatias de `size` chars com `overlap` de sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# -------------------------------
# MELHORIA 1 — Cache de embeddings
# -------------------------------


def _corpus_fingerprint(file_names: list[str], base_dir: str) -> str:
    """Hash SHA-256 dos mtimes + tamanhos dos arquivos do corpus."""
    h = hashlib.sha256()
    for fname in sorted(file_names):
        path = os.path.join(base_dir, fname)
        stat = os.stat(path)
        h.update(f"{fname}:{stat.st_mtime}:{stat.st_size}".encode())
    return h.hexdigest()


def load_corpus(transcriptions_dir: str):
    """
    Carrega chunks e embeddings do corpus.
    Usa cache em disco se o corpus não mudou desde a última execução.
    """
    md_files = sorted(Path(transcriptions_dir).glob("*.md"))
    if not md_files:
        log.error(
            "Nenhum arquivo .md encontrado em '%s'. Verifique o diretório.",
            transcriptions_dir,
        )
        sys.exit(1)

    file_names = [f.name for f in md_files]
    fingerprint = _corpus_fingerprint(file_names, transcriptions_dir)

    # Tenta carregar cache
    if os.path.exists(CACHE_PATH) and os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH, "r") as f:
            meta = json.load(f)
        if meta.get("fingerprint") == fingerprint:
            log.info("Cache de embeddings válido — carregando do disco...")
            cached = torch.load(CACHE_PATH, map_location=DEVICE)
            log.info(
                "%d arquivos, %d chunks carregados do cache.",
                len(file_names),
                len(cached["docs"]),
            )
            return cached["docs"], cached["file_index"], cached["embeddings"]

    # Reconstrói corpus e embeddings
    log.info("Processando transcrições (cache desatualizado ou inexistente)...")
    docs: list[str] = []
    file_index: list[str] = []

    for fname in file_names:
        path = os.path.join(transcriptions_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        docs.extend(chunks)
        file_index.extend([fname] * len(chunks))

    log.info("Gerando embeddings para %d chunks...", len(docs))
    embeddings = embed_model.encode(
        docs, convert_to_tensor=True, device=DEVICE, show_progress_bar=True
    )

    # Persiste cache
    torch.save(
        {"docs": docs, "file_index": file_index, "embeddings": embeddings}, CACHE_PATH
    )
    with open(CACHE_META_PATH, "w") as f:
        json.dump({"fingerprint": fingerprint}, f)
    log.info("Cache salvo em '%s'.", CACHE_PATH)

    return docs, file_index, embeddings


# -------------------------------
# MELHORIA 5 — Contagem de tokens real
# -------------------------------


def count_tokens(text: str) -> int:
    """Usa o tokenizador do LLM para contar tokens com precisão."""
    return len(llm.tokenize(text.encode("utf-8")))


# -------------------------------
# Retriever + MELHORIA 8 — Re-ranking
# -------------------------------


def retrieve_chunks(
    query: str,
    docs: list[str],
    file_index: list[str],
    embeddings: torch.Tensor,
    selected_files: list[str] | None = None,
) -> list[dict]:
    """
    1. Busca semântica (bi-encoder) → TOP_K_RETRIEVER candidatos
    2. Re-ranking com Cross-Encoder → TOP_K_FINAL melhores
    3. Filtra pelo limite de tokens do contexto
    """
    with torch.no_grad():
        query_emb = embed_model.encode(query, convert_to_tensor=True, device=DEVICE)
        scores = util.cos_sim(query_emb, embeddings).squeeze(0).cpu()

    if selected_files:
        candidate_indices = [i for i, f in enumerate(file_index) if f in selected_files]
        candidate_scores = scores[candidate_indices]
        k = min(TOP_K_RETRIEVER, len(candidate_indices))
        top_local = torch.topk(candidate_scores, k=k).indices.tolist()
        top_indices = [candidate_indices[i] for i in top_local]
    else:
        k = min(TOP_K_RETRIEVER, len(docs))
        top_indices = torch.topk(scores, k=k).indices.tolist()

    # Re-ranking com Cross-Encoder
    pairs = [(query, docs[i]) for i in top_indices]
    ce_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(ce_scores, top_indices), key=lambda x: x[0], reverse=True)
    top_reranked = [idx for _, idx in ranked[:TOP_K_FINAL]]

    # Respeita limite de tokens
    selected: list[dict] = []
    tokens_so_far = 0
    for i in top_reranked:
        chunk = docs[i]
        t = count_tokens(chunk)
        if tokens_so_far + t > MAX_CONTEXT_TOKENS:
            break
        selected.append({"text": chunk, "file": file_index[i]})
        tokens_so_far += t

    return selected


# -------------------------------
# MELHORIA 6 + 9 — Prompt com delimitadores e instrução de citação
# -------------------------------


def build_prompt(chunks: list[dict], query: str, mode: str) -> str:
    # Delimitadores fortes entre chunks
    chunk_block = "\n\n".join(
        f'<doc id="{i + 1}" file="{c["file"]}">\n{c["text"]}\n</doc>'
        for i, c in enumerate(chunks)
    )

    cite_instruction = (
        "Ao responder, cite o arquivo de origem usando o atributo 'file' das tags <doc>, "
        "por exemplo: (fonte: arquivo.md)."
    )

    if mode == "trecho":
        return (
            f"Você é um assistente. Retorne os trechos relevantes para a pergunta. {cite_instruction}\n\n"
            f"{chunk_block}\n\n"
            f"Pergunta: {query}\nResposta:"
        )
    elif mode == "resumo":
        return (
            f"Você é um assistente. Resuma os documentos abaixo de forma clara. {cite_instruction}\n\n"
            f"{chunk_block}\n\n"
            f"Resumo:"
        )
    elif mode == "explicacao":
        return (
            f"Você é um assistente. Explique os documentos abaixo de forma detalhada. {cite_instruction}\n\n"
            f"{chunk_block}\n\n"
            f"Explicação:"
        )
    else:
        return f"{chunk_block}\n\nPergunta: {query}\nResposta:"


def ask_llm(chunks: list[dict], query: str, mode: str = "trecho") -> str:
    prompt = build_prompt(chunks, query, mode)
    try:
        resp = llm(prompt=prompt, max_tokens=400, temperature=0.2)
        return resp["choices"][0]["text"].strip()
    except ValueError as exc:
        log.error("Prompt excedeu capacidade do modelo: %s", exc)
        return "❌ Erro: prompt muito longo. Tente uma pergunta menor."


# -------------------------------
# MELHORIA 4 — Seleção de arquivo dentro do loop
# -------------------------------


def select_files(file_names: list[str]) -> list[str] | None:
    print("\nTranscrições disponíveis:\n")
    for i, fname in enumerate(file_names, 1):
        print(f"  {i} - {fname}")
    print("  0 - Buscar em todas")

    raw = input("\nEscolha (número ou 0 para todas): ").strip()
    try:
        choice = int(raw)
    except ValueError:
        log.warning("Entrada inválida, buscando em todas.")
        return None
    return None if choice == 0 else [file_names[choice - 1]]


# -------------------------------
# Loop Principal
# -------------------------------


def main():
    docs, file_index, embeddings = load_corpus(TRANSCRIPTIONS_DIR)
    file_names = sorted(set(file_index))

    log.info(
        "Corpus pronto: %d arquivos, %d chunks.",
        len(file_names),
        len(docs),
    )

    selected_files = select_files(file_names)

    while True:
        query = input("\nPergunta (ou 'sair' / 'mudar arquivo'): ").strip()

        if query.lower() == "sair":
            break

        # MELHORIA 4 — troca de escopo sem reiniciar
        if query.lower() == "mudar arquivo":
            selected_files = select_files(file_names)
            continue

        if not query:
            continue

        mode = input("Modo (trecho/resumo/explicacao) [trecho]: ").strip().lower()
        if mode not in ("trecho", "resumo", "explicacao"):
            mode = "trecho"

        chunks = retrieve_chunks(query, docs, file_index, embeddings, selected_files)

        # MELHORIA 10 — guarda para corpus vazio / sem resultados
        if not chunks:
            log.warning("Nenhum conteúdo relevante encontrado para essa pergunta.")
            continue

        log.info(
            "%d chunks recuperados e re-rankeados de: %s",
            len(chunks),
            ", ".join(set(c["file"] for c in chunks)),
        )

        answer = ask_llm(chunks, query, mode)
        print(f"\n💬 {mode.capitalize()}:\n{answer}\n")


if __name__ == "__main__":
    main()
