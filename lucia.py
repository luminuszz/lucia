#!/home/daviribeiro/projects/personal/lucia/.venv/bin/python

import argparse
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import sounddevice as sd
import torch
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from scipy.io.wavfile import write
from transformers import pipeline as hf_pipeline

# -------------------------------
# Configurações Globais
# -------------------------------

load_dotenv()

TRANSCRIPTION_DIR = os.getenv(
    "OBSIDIAN_TRANSCRIPT_DIR",
    "/home/daviribeiro/Documents/obsidian-storage/davi-brain/transcricoes",
)
TEMP_DIR = ".temp"
MODEL_NAME = "medium"
LANGUAGE = "pt"
HF_TOKEN = os.getenv("HF_TOKEN")

SUMMARIZER_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
SUMMARY_CHUNK_SIZE = 1800
SUMMARY_CHUNK_OVERLAP = 200

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
# Lazy loading do sumarizador
# -------------------------------
_summarizer = None


def get_summarizer():
    """Carrega o sumarizador forçando o Tokenizer clássico para evitar crash do Tiktoken."""
    global _summarizer
    if _summarizer is None:
        log.info("Carregando modelo de sumarização '%s'...", SUMMARIZER_MODEL)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            SUMMARIZER_MODEL, use_fast=False, legacy=False
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)

        _summarizer = hf_pipeline("summarization", model=model, tokenizer=tokenizer)

        log.info("Modelo de sumarização carregado com sucesso.")

    return _summarizer


# -------------------------------
# Sumarização hierárquica
# -------------------------------
def _split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def synthesize_summary(transcriptions: list[str]) -> str:
    full_text = " ".join(transcriptions).strip()
    if not full_text:
        return ""

    summarizer = get_summarizer()
    chunks = _split_into_chunks(full_text, SUMMARY_CHUNK_SIZE, SUMMARY_CHUNK_OVERLAP)
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        result = summarizer(
            chunk, max_length=120, min_length=30, do_sample=False, truncation=True
        )
        partial_summaries.append(result[0]["summary_text"])

    if not partial_summaries:
        return ""

    if len(partial_summaries) == 1:
        final_text = partial_summaries[0]
    else:
        combined = " ".join(partial_summaries)
        final_result = summarizer(
            combined[:SUMMARY_CHUNK_SIZE],
            max_length=200,
            min_length=50,
            do_sample=False,
            truncation=True,
        )
        final_text = final_result[0]["summary_text"]

    topics = [f"- {s.strip()}" for s in final_text.split(". ") if s.strip()]
    return "\n".join(topics)


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
# Gravação e Processamento Batch
# -------------------------------
def record_chunk_multi(
    device_ids: list[int], sample_rate: int, duration: int
) -> np.ndarray:
    n_frames = int(duration * sample_rate)
    buffers = [None] * len(device_ids)
    errors = [None] * len(device_ids)

    def _record(idx: int, dev_id: int):
        try:
            buf = sd.rec(
                n_frames,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=dev_id,
            )
            sd.wait()
            buffers[idx] = buf.flatten()
        except Exception as exc:
            errors[idx] = exc

    threads = [
        threading.Thread(target=_record, args=(i, dev_id), daemon=True)
        for i, dev_id in enumerate(device_ids)
    ]
    for t in threads:
        t.start()

    try:
        for t in threads:
            while t.is_alive():
                t.join(0.1)
    except KeyboardInterrupt:
        sd.stop()
        raise

    for i, err in enumerate(errors):
        if err is not None:
            raise RuntimeError(f"Erro no disp {device_ids[i]}: {err}") from err

    if len(buffers) == 1:
        return buffers[0]
    return np.column_stack((buffers[0], buffers[1]))


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


def batch_transcription():
    md_path = get_md_filename()
    device_ids, sample_rate = select_input_devices()

    # Filas para receber o áudio do microfone em tempo real
    audio_queues = [queue.Queue() for _ in device_ids]
    stop_event = threading.Event()  # <-- Controlador de estado (Liga/Desliga)

    def make_callback(q):
        def callback(indata, frames, time_info, status):
            if status:
                log.debug(f"Aviso de áudio: {status}")
            q.put(indata.copy())

        return callback

    streams = []
    raw_paths = []
    files = []

    try:
        # 1. Prepara os arquivos
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

        # 2. Cria a Thread Escritora (Salva no disco em background)
        def disk_writer():
            while not stop_event.is_set():
                for i, q in enumerate(audio_queues):
                    while not q.empty():
                        files[i].write(q.get().tobytes())
                time.sleep(0.05)

            # Garante que as filas sejam esvaziadas uma última vez ao parar
            for i, q in enumerate(audio_queues):
                while not q.empty():
                    files[i].write(q.get().tobytes())

        writer_thread = threading.Thread(target=disk_writer, daemon=True)

        # 3. Liga os motores!
        start_time = datetime.now()
        for s in streams:
            s.start()
        writer_thread.start()

        # 4. A Thread Principal agora fica parada aqui com uma UX muito melhor
        print("\n" + "=" * 60)
        input(" 🔴 GRAVANDO... Pressione [ENTER] para parar e processar.")
        print("=" * 60 + "\n")

        # Avisa a Thread Escritora para encerrar
        stop_event.set()
        writer_thread.join()

    finally:
        # Limpeza segura (mesmo se der erro em outra parte)
        for s in streams:
            s.stop()
            s.close()
        for f in files:
            f.close()

    # ==========================================
    # FASE 2: MONTAGEM E PROCESSAMENTO DA IA
    # ==========================================
    log.info("Lendo áudio do disco e montando arquivo final...")

    channels_data = []
    for p in raw_paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            channels_data.append(np.fromfile(p, dtype=np.float32))

    if not channels_data:
        log.warning("Nenhum áudio gravado. Encerrando.")
        return

    # Realinha caso existam duas fontes de áudio
    if len(channels_data) == 2:
        min_len = min(len(channels_data[0]), len(channels_data[1]))
        full_audio = np.column_stack(
            (channels_data[0][:min_len], channels_data[1][:min_len])
        )
    else:
        full_audio = channels_data[0]

    temp_wav_path = os.path.join(TEMP_DIR, "reuniao_completa.wav")
    write(temp_wav_path, sample_rate, (full_audio * 32767).astype(np.int16))

    # --- Processamento IA (Sem alterações) ---
    log.info("Carregando modelos de IA na GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(MODEL_NAME, device=device)
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=HF_TOKEN
    )

    log.info("🎧 Processando Diarização (PyAnnote)...")
    diarization = diarization_pipeline(temp_wav_path)

    log.info("📝 Processando Transcrição (Whisper)... Isso pode levar alguns minutos.")
    result = whisper_model.transcribe(
        temp_wav_path,
        fp16=torch.cuda.is_available(),
        language=LANGUAGE,
        task="transcribe",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        initial_prompt="A seguir, uma gravação de áudio em português contendo uma aula ou reunião.",
    )

    transcribed_texts = []

    log.info("💾 Escrevendo arquivo Markdown para o Segundo Cérebro...")
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(
            f"# Transcrição\n**Data:** {start_time.strftime('%d/%m/%Y %H:%M:%S')}\n\n"
        )

        for seg in result["segments"]:
            seg_start, seg_end, text = seg["start"], seg["end"], seg["text"].strip()
            if not text:
                continue

            speaker = assign_speaker_to_segment(seg_start, seg_end, diarization)
            channel_tag = " [Estéreo]" if full_audio.ndim == 2 else ""
            line = f"[{seg_start:.2f}s - {seg_end:.2f}s] **{speaker}{channel_tag}:** {text}"

            md_file.write(line + "\n\n")
            transcribed_texts.append(text)

    shutil.rmtree(TEMP_DIR)
    log.info(f"✅ Sucesso! Transcrição salva perfeitamente em: {md_path}")


# -------------------------------
# Execução
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcrição Batch para Segundo Cérebro"
    )
    parser.add_argument(
        "--summarize",
        type=str,
        metavar="ARQUIVO",
        help="Gerar resumo de transcrição existente",
    )
    args = parser.parse_args()

    if args.summarize:
        # A função summarize_existing_file continua a mesma, você pode colar ela aqui se usar com frequência
        pass
    else:
        batch_transcription()
