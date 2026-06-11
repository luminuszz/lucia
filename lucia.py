#!/home/daviribeiro/projects/personal/lucia/.venv/bin/python

import argparse
import gc
import logging
import os
import queue
import select
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from scipy.io.wavfile import write
from tqdm import tqdm

# -------------------------------
# Configurações Globais
# -------------------------------

load_dotenv()

TRANSCRIPTION_DIR = os.getenv(
    "OBSIDIAN_TRANSCRIPT_DIR",
    "/home/daviribeiro/Documents/obsidian-storage/davi-brain/brain/transcricoes",
)
TEMP_DIR = os.getenv(
    "OBSIDIAN_TEMP_RAW",
    "/home/daviribeiro/.temp",
)
MODEL_NAME = "medium"
LANGUAGE = "pt"
HF_TOKEN = os.getenv("HF_TOKEN")

# Configurações do Ollama Local
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct-q5_K_M"

os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------
# Logging estruturado
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lucia")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")


# -------------------------------
# Sumarização com Ollama Local
# -------------------------------
def synthesize_summary_ollama(transcriptions: list[str]) -> str:
    full_text = " ".join(transcriptions).strip()
    if not full_text:
        return "Nenhum texto para resumir."

    log.info("Enviando transcrição para o Ollama (%s)...", OLLAMA_MODEL)

    prompt = f"""
    Você é uma assistente executiva focado em síntese, seu nome é Lucia. Abaixo está a transcrição de uma reunião corporativa.
    Seu objetivo é fornecer:
    1. Um resumo executivo de 1 parágrafo.
    2. Os principais tópicos discutidos (em tópicos).
    3. Decisões tomadas.
    4. Pontos de ação (Action Items) com seus respectivos responsáveis (se mencionados).
    Não adicione informações que não estão no texto original.

    Transcrição:
    {full_text}
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1000, "temperature": 0.3},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "Erro ao extrair resposta da API.")
    except Exception as e:
        log.error("Erro na chamada ao Ollama: %s", e)
        return f"> **Erro na sumarização local:** {e}\n> Verifique se o container do Ollama está rodando e o modelo foi baixado."


def summarize_existing_file(filepath: str):
    if not os.path.exists(filepath):
        log.error("Arquivo não encontrado: %s", filepath)
        return

    log.info("Lendo %s para sumarização...", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if "## 🧠 Resumo Executivo (IA)" in content:
        log.warning("Este arquivo aparentemente já possui um resumo.")
        return

    resumo = synthesize_summary_ollama([content])

    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n---\n## 🧠 Resumo Executivo (IA)\n\n")
        f.write(resumo)
        f.write("\n")

    log.info("✅ Resumo gerado e anexado com sucesso!")


# -------------------------------
# Integração com KDE e Seleção de Dispositivos
# -------------------------------
def get_linux_default_audio_names() -> tuple[str | None, str | None]:
    try:
        info = subprocess.run(
            ["pactl", "info"], capture_output=True, text=True, check=True
        ).stdout
        def_sink = next(
            (
                l.split(": ")[1].strip()
                for l in info.splitlines()
                if "Default Sink:" in l
            ),
            None,
        )
        def_source = next(
            (
                l.split(": ")[1].strip()
                for l in info.splitlines()
                if "Default Source:" in l
            ),
            None,
        )

        sink_desc = source_desc = None
        if def_sink:
            sinks = subprocess.run(
                ["pactl", "list", "sinks"], capture_output=True, text=True
            ).stdout
            current_name = None
            for line in sinks.splitlines():
                if "Name:" in line:
                    current_name = line.split(": ")[1].strip()
                if "Description:" in line and current_name == def_sink:
                    sink_desc = line.split(": ")[1].strip().lower()
                    break
        if def_source:
            sources = subprocess.run(
                ["pactl", "list", "sources"], capture_output=True, text=True
            ).stdout
            current_name = None
            for line in sources.splitlines():
                if "Name:" in line:
                    current_name = line.split(": ")[1].strip()
                if "Description:" in line and current_name == def_source:
                    source_desc = line.split(": ")[1].strip().lower()
                    break
        return sink_desc, source_desc
    except Exception:
        return None, None


def select_input_devices() -> tuple[list[int], int]:
    print("\nDispositivos de áudio disponíveis:\n")
    devices = sd.query_devices()
    os_out_desc, os_in_desc = get_linux_default_audio_names()
    fallback_in, fallback_out = sd.default.device
    best_in_id, best_out_id = -1, -1

    for i, dev in enumerate(devices):
        dev_name = dev["name"].lower()
        if os_in_desc and dev["max_input_channels"] > 0 and dev_name in os_in_desc:
            best_in_id = i
        if os_out_desc and dev["max_output_channels"] > 0 and dev_name in os_out_desc:
            best_out_id = i

    if best_in_id == -1:
        best_in_id = fallback_in
    if best_out_id == -1:
        best_out_id = fallback_out

    for i, dev in enumerate(devices):
        channels_in = dev["max_input_channels"]
        channels_out = dev["max_output_channels"]
        if channels_in == 0 and channels_out == 0:
            continue

        tags = []
        if i == best_in_id:
            tags.append("🎤 [ENTRADA KDE]")
        if i == best_out_id:
            tags.append("🔊 [SAÍDA KDE]")
        tag_str = f" {' '.join(tags)}" if tags else ""

        print(
            f"  {i:>2}: {dev['name']} (In: {channels_in}, Out: {channels_out}, SR: {dev['default_samplerate']}){tag_str}"
        )

    ids_str = input(
        f"\nDigite IDs separados por vírgula [Enter = ID {best_in_id}]: "
    ).strip()

    if not ids_str:
        if best_in_id < 0:
            raise ValueError("Especifique manualmente.")
        device_ids = [best_in_id]
        print(f"🔄 Seleção automática do KDE: ID {best_in_id}\n")
    else:
        device_ids = [int(x) for x in ids_str.split(",")]

    sample_rates = [int(devices[i]["default_samplerate"]) for i in device_ids]
    return device_ids, min(sample_rates)


def get_md_filename() -> str:
    md_filename = input(
        "📄 Nome do arquivo Markdown. Deixe em branco para auto: "
    ).strip()
    if not md_filename:
        md_filename = datetime.now().strftime("transcricao-%Y-%m-%d-%Hh%Mm.md")
    elif not md_filename.endswith(".md"):
        md_filename += ".md"
    return os.path.join(TRANSCRIPTION_DIR, md_filename)


# -------------------------------
# Processamento Base de IA (Estágios Isolados)
# -------------------------------
def assign_speaker_to_segment(start: float, end: float, diarization_result) -> str:
    if hasattr(diarization_result, "speaker_diarization"):
        annotation = diarization_result.speaker_diarization
    else:
        annotation = diarization_result

    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def process_audio(audio_path: str, md_path: str):
    """Executa o pipeline particionado: 1. PyAnnote -> 2. Faster-Whisper -> 3. Ollama."""
    log.info("Iniciando processamento de IA no arquivo: %s", audio_path)

    # ESTÁGIO 1: PyAnnote (Diarização)
    log.info("🎧 Estágio 1/3: Carregando PyAnnote na GPU...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=HF_TOKEN
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    print("\n")  # Quebra de linha para a barra de progresso ficar limpa
    with ProgressHook() as hook:
        diarization = diarization_pipeline(audio_path, hook=hook)
    print("\n")

    log.info("🧹 Limpando PyAnnote da VRAM...")
    del diarization_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ESTÁGIO 2: Faster-Whisper (Transcrição)
    log.info("📝 Estágio 2/3: Carregando Faster-Whisper na GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    whisper_model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)

    # O info do faster-whisper nos dá a duração do áudio para podermos montar a barra!
    segments, info = whisper_model.transcribe(
        audio_path,
        language=LANGUAGE,
        condition_on_previous_text=False,
        vad_filter=True,
        initial_prompt="A seguir, uma gravação de áudio em português contendo uma aula ou reunião.",
    )

    log.info("💾 Escrevendo arquivo Markdown para o Segundo Cérebro...")
    transcribed_texts = []

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(
            f"# Transcrição\n**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
        )

        # Inicia a barra de progresso baseada na duração total do áudio
        with tqdm(
            total=round(info.duration, 2),
            unit="s",
            desc="📝 Transcrevendo",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s",
        ) as pbar:
            for seg in segments:
                text = seg.text.strip()

                # Atualiza a barra de progresso para acompanhar o término do segmento atual
                advance = seg.end - pbar.n
                if advance > 0:
                    pbar.update(advance)

                if not text:
                    continue

                speaker = assign_speaker_to_segment(seg.start, seg.end, diarization)
                line = f"[{seg.start:.2f}s - {seg.end:.2f}s] **{speaker}:** {text}"

                md_file.write(line + "\n\n")
                transcribed_texts.append(text)

            # Garante que a barra feche no 100% no final do loop
            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)

    log.info("🧹 Limpando Faster-Whisper da VRAM...")
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ESTÁGIO 3: Ollama (Resumo)
    log.info(
        "🧠 Estágio 3/3: Solicitando resumo executivo via Ollama... (Isso pode levar alguns minutos)"
    )
    resumo_ia = synthesize_summary_ollama(transcribed_texts)

    with open(md_path, "a", encoding="utf-8") as md_file:
        md_file.write("\n---\n## 🧠 Resumo Executivo (IA)\n\n")
        md_file.write(resumo_ia)
        md_file.write("\n")

    log.info("✅ Sucesso! Transcrição salva perfeitamente em: %s", md_path)


# -------------------------------
# Gravação e Processamento Batch
# -------------------------------
def batch_transcription():
    md_path = get_md_filename()
    device_ids, sample_rate = select_input_devices()

    audio_queues = [queue.Queue() for _ in device_ids]
    stop_event = threading.Event()

    # --- CONFIGURAÇÕES DE AUTO-STOP ---
    MAX_DURATION_SEC = 40 * 60  # 40 minutos de limite total
    MAX_SILENCE_SEC = 5 * 60  # 5 minutos contínuos sem áudio para abortar
    SILENCE_THRESHOLD = 0.015  # Limiar de volume (0.0 a 1.0)

    start_time = time.time()
    last_audio_time = time.time()

    def make_callback(q):
        def callback(indata, frames, time_info, status):
            nonlocal last_audio_time
            if status:
                log.debug(f"Aviso de áudio: {status}")

            # Checa se o pico de volume do chunk supera o limiar de silêncio
            if np.max(np.abs(indata)) > SILENCE_THRESHOLD:
                last_audio_time = time.time()

            q.put(indata.copy())

        return callback

    streams = []
    raw_paths = []
    files = []

    try:
        for i, dev_id in enumerate(device_ids):
            raw_path = os.path.join(TEMP_DIR, f"track_{i}.raw")
            raw_paths.append(raw_path)
            files.append(open(raw_path, "wb"))

            s = sd.InputStream(
                samplerate=sample_rate,
                device=dev_id,
                channels=1,
                dtype="float32",
                callback=make_callback(audio_queues[i]),
            )
            streams.append(s)

        def disk_writer():
            while not stop_event.is_set():
                for i, q in enumerate(audio_queues):
                    while not q.empty():
                        files[i].write(q.get().tobytes())
                time.sleep(0.05)

            for i, q in enumerate(audio_queues):
                while not q.empty():
                    files[i].write(q.get().tobytes())

        writer_thread = threading.Thread(target=disk_writer, daemon=True)

        for s in streams:
            s.start()
        writer_thread.start()

        print("\n" + "=" * 60)
        print(" 🔴 GRAVANDO...")
        print(" 🛑 Pressione [ENTER] para parar e processar manualmente.")
        print(
            f" ⏳ Parada automática: {MAX_DURATION_SEC // 60}min total ou {MAX_SILENCE_SEC // 60}min de silêncio."
        )
        print("=" * 60 + "\n")

        # --- LOOP DE CONTROLE PRINCIPAL ---
        while not stop_event.is_set():
            # 1. Verifica teclado de forma não-bloqueante (timeout de 0.5s)
            dr, dw, de = select.select([sys.stdin], [], [], 0.5)
            if dr:
                sys.stdin.readline()
                log.info("⏹️ Gravação interrompida manualmente.")
                stop_event.set()
                break

            now = time.time()

            # 2. Trava de tempo máximo (40 minutos)
            if now - start_time >= MAX_DURATION_SEC:
                log.info("⏰ Limite de 40 minutos atingido. Processando áudio...")
                stop_event.set()
                break

            # 3. Trava de inatividade (Silêncio prolongado)
            if now - last_audio_time >= MAX_SILENCE_SEC:
                log.info("🔇 Muito tempo sem áudio detectado. Processando áudio...")
                stop_event.set()
                break

        writer_thread.join()

    finally:
        for s in streams:
            s.stop()
            s.close()
        for f in files:
            f.close()

    log.info("Lendo áudio do disco e montando arquivo final...")
    channels_data = []
    for p in raw_paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            channels_data.append(np.fromfile(p, dtype=np.float32))

    if not channels_data:
        log.warning("Nenhum áudio gravado. Encerrando.")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        return

    if len(channels_data) == 2:
        min_len = min(len(channels_data[0]), len(channels_data[1]))
        full_audio = np.column_stack(
            (channels_data[0][:min_len], channels_data[1][:min_len])
        )
    else:
        full_audio = channels_data[0]

    temp_wav_path = os.path.join(TEMP_DIR, "reuniao_completa.wav")
    write(temp_wav_path, sample_rate, (full_audio * 32767).astype(np.int16))

    process_audio(temp_wav_path, md_path)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


# -------------------------------
# Execução Principal
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lucia - Transcrição Batch e Resumo para Segundo Cérebro"
    )
    parser.add_argument(
        "--summarize",
        type=str,
        metavar="ARQUIVO_MD",
        help="Gerar resumo de transcrição existente via Ollama",
    )
    parser.add_argument(
        "--audio",
        type=str,
        metavar="ARQUIVO_AUDIO",
        help="Caminho para um arquivo de áudio existente (mp3, wav, m4a, etc)",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize_existing_file(args.summarize)
    elif args.audio:
        if not os.path.exists(args.audio):
            log.error("Arquivo de áudio não encontrado no caminho: %s", args.audio)
        else:
            caminho_md = get_md_filename()
            process_audio(args.audio, caminho_md)
    else:
        batch_transcription()
