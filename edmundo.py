#!/home/daviribeiro/projects/personal/lucia/.venv/bin/python

import os
import re
import json
import logging
import sys
import requests
import torch
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# -------------------------------
# Configurações Globais
# -------------------------------

load_dotenv()

TRANSCRIPTIONS_DIR = os.getenv(
    "OBSIDIAN_TRANSCRIPT_DIR",
    "/home/daviribeiro/Documents/obsidian-storage/davi-brain/brain/transcricoes",
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral-nemo"
EMBED_MODEL_NAME = "all-mpnet-base-v2"

# Cache local para não precisar re-indexar tudo toda hora
CACHE_FILE = "/home/daviribeiro/projects/personal/lucia/.edmundo_embeddings.pt"
CACHE_META = "/home/daviribeiro/projects/personal/lucia/.edmundo_meta.json"

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
# Modelos
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Usando dispositivo: {DEVICE}")

log.info(f"Carregando modelo de embeddings: {EMBED_MODEL_NAME}...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

# -------------------------------
# Funções de Suporte
# -------------------------------

def parse_transcription_file(file_path):
    """
    Analisa o arquivo markdown e extrai os blocos de transcrição com timestamps.
    Retorna uma lista de dicionários: {'file', 'timestamp', 'speaker', 'text'}
    """
    fname = os.path.basename(file_path)
    entries = []
    
    # Regex para capturar: [timestamp opcional] [timestamp relativo] **SPEAKER:** texto
    # Exemplo: [11:44:32 - 11:45:02] [0.00-1.24] UNKNOWN: Progride análise médica.
    # Exemplo: [0.00s - 1.22s] **SPEAKER_00:** text
    
    # Esta regex tenta ser flexível para ambos os formatos
    pattern = re.compile(r'^(?:\[.*?\]\s*)?(\[.*?\])\s*\**([^*:]+?)\**:\s*(.*)$')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('---') or line.startswith('**Data:**'):
                    continue
                
                # Se a linha for o início de um resumo, paramos de ler a transcrição pura
                if '## 🧠 Resumo Executivo (IA)' in line:
                    break
                    
                match = pattern.match(line)
                if match:
                    timestamp = match.group(1).strip()
                    speaker = match.group(2).strip()
                    text = match.group(3).strip()
                    
                    entries.append({
                        'file': fname,
                        'timestamp': timestamp,
                        'speaker': speaker,
                        'text': text
                    })
    except Exception as e:
        log.error(f"Erro ao ler {file_path}: {e}")
        
    return entries

def get_corpus_fingerprint(files):
    import hashlib
    h = hashlib.sha256()
    for f in sorted(files):
        try:
            st = os.stat(f)
            h.update(f"{os.path.basename(f)}:{st.st_mtime}:{st.st_size}".encode())
        except Exception:
            continue
    return h.hexdigest()

def load_and_index():
    md_files = list(Path(TRANSCRIPTIONS_DIR).glob("*.md"))
    if not md_files:
        log.error(f"Nenhum arquivo encontrado em {TRANSCRIPTIONS_DIR}")
        return [], None

    fingerprint = get_corpus_fingerprint([str(f) for f in md_files])
    
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_META):
        try:
            with open(CACHE_META, 'r') as f:
                meta = json.load(f)
            if meta.get('fingerprint') == fingerprint:
                log.info("Cache válido - carregando embeddings...")
                cache = torch.load(CACHE_FILE, map_location=DEVICE, weights_only=False)
                return cache['entries'], cache['embeddings']
        except Exception as e:
            log.warning(f"Erro ao carregar cache: {e}. Re-indexando...")

    log.info("Processando transcrições para indexação...")
    all_entries = []
    for f in md_files:
        all_entries.extend(parse_transcription_file(f))
    
    if not all_entries:
        log.warning("Nenhuma entrada de transcrição válida encontrada.")
        return [], None

    texts = [f"{e['speaker']}: {e['text']}" for e in all_entries]
    log.info(f"Gerando embeddings para {len(texts)} segmentos...")
    embeddings = embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    try:
        torch.save({'entries': all_entries, 'embeddings': embeddings}, CACHE_FILE)
        with open(CACHE_META, 'w') as f:
            json.dump({'fingerprint': fingerprint}, f)
        log.info("Cache de indexação salvo com sucesso.")
    except Exception as e:
        log.error(f"Erro ao salvar cache: {e}")
        
    return all_entries, embeddings

def ask_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1000}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "Erro ao extrair resposta da API.")
    except Exception as e:
        return f"❌ Erro na conexão com Ollama: {e}"

def search(query, entries, embeddings, top_k=15, selected_file=None):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    
    if selected_file:
        indices = [i for i, e in enumerate(entries) if e['file'] == selected_file]
        if not indices:
            return []
        sub_embeddings = embeddings[indices]
        cos_scores = util.cos_sim(query_emb, sub_embeddings)[0]
        # Garantir que k não é maior que o número de elementos
        k = min(top_k, len(indices))
        top_results = torch.topk(cos_scores, k=k)
        return [entries[indices[i]] for i in top_results.indices]
    else:
        cos_scores = util.cos_sim(query_emb, embeddings)[0]
        k = min(top_k, len(entries))
        top_results = torch.topk(cos_scores, k=k)
        return [entries[i] for i in top_results.indices]

# -------------------------------
# Loop Principal
# -------------------------------

def main():
    entries, embeddings = load_and_index()
    if not entries:
        log.error("Encerrando: Corpus vazio.")
        return

    files = sorted(list(set(e['file'] for e in entries)))
    current_file = None

    print("\n" + "═"*60)
    print(" 🧠  EDMUNDO - SEU SEGUNDO CÉREBRO (OLLAMA EDITION)")
    print(" 📂  Base: " + TRANSCRIPTIONS_DIR)
    print(" 📝  Segmentos indexados: " + str(len(entries)))
    print("═"*60)

    while True:
        scope = f"📄 {current_file}" if current_file else "🌐 Todas as transcrições"
        print(f"\n[Escopo: {scope}]")
        print("Comandos: 'mudar' (selecionar arquivo), 'todas' (voltar para busca geral), 'sair'")
        
        query = input("\nPergunte algo: ").strip()

        if not query:
            continue
            
        if query.lower() == 'sair':
            print("Até logo!")
            break
        elif query.lower() == 'mudar':
            print("\nArquivos disponíveis:")
            for i, f in enumerate(files):
                print(f"  {i+1}. {f}")
            choice = input("\nEscolha o número (ou Enter para cancelar): ")
            if choice.isdigit() and 1 <= int(choice) <= len(files):
                current_file = files[int(choice)-1]
                print(f"✅ Escopo alterado para: {current_file}")
            continue
        elif query.lower() == 'todas':
            current_file = None
            print("✅ Escopo alterado para: Todas as transcrições")
            continue

        log.info(f"Buscando contexto relevante para: '{query}'...")
        results = search(query, entries, embeddings, selected_file=current_file)

        if not results:
            print("⚠️ Nenhuma informação relevante encontrada.")
            continue

        # Montar contexto para o LLM
        context_blocks = []
        for i, res in enumerate(results):
            block = f"--- FONTE {i+1} ---\n"
            block += f"ARQUIVO: {res['file']}\n"
            block += f"TIMESTAMP: {res['timestamp']}\n"
            block += f"QUEM FALOU: {res['speaker']}\n"
            block += f"FALA: {res['text']}"
            context_blocks.append(block)
        
        context_text = "\n\n".join(context_blocks)

        prompt = f"""
Você é o Edmundo, um assistente especializado em analisar transcrições de reuniões e aulas.
Seu objetivo é responder perguntas do usuário com base EXCLUSIVAMENTE nos trechos fornecidos abaixo.

DIRETRIZES DE RESPOSTA:
1. Responda de forma clara e objetiva.
2. Seja fiel aos fatos relatados nos trechos.
3. SEMPRE cite de qual arquivo e timestamp a informação foi extraída. 
   Exemplo: "Segundo fulano, o projeto atrasou (Arquivo: reuniao.md, Timestamp: [10:00-10:15])"
4. Se a resposta não puder ser encontrada nos trechos, diga: "Infelizmente não encontrei essa informação nas minhas transcrições."
5. Responda em Português do Brasil.

CONTEXTO RELEVANTE:
{context_text}

PERGUNTA:
{query}

RESPOSTA:
"""

        log.info("Consultando Ollama...")
        answer = ask_ollama(prompt)
        
        print("\n" + "─"*60)
        print(f"💡 RESPOSTA DO EDMUNDO:\n\n{answer}")
        print("─"*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Saindo...")
        sys.exit(0)
