#!/home/daviribeiro/projects/personal/lucia/venv/bin/python
import dotenv

import os
import sys
import warnings
import shutil
import argparse
from datetime import datetime, timedelta
import sounddevice as sd
import numpy as np
import torch
import whisper
from scipy.io.wavfile import write
from pyannote.audio import Pipeline
from transformers import pipeline  # sumarização

# -------------------------------
# Configurações Globais
# -------------------------------

dotenv.load_dotenv()

CHUNK_DURATION = 20        # segundos por chunk de gravação
TRANSCRIPTION_DIR = "/home/daviribeiro/Documents/obsidian-storage/davi-brain/transcricoes"
TEMP_DIR = ".temp"
MODEL_NAME = "medium" 
LANGUAGE = "pt"
HF_TOKEN = os.getenv("HF_TOKEN")

# Criar diretórios se não existirem
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------
# Silenciar warnings chatos
# -------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*Model was trained with.*")

# -------------------------------
# Sumarização (local, gratuita)
# -------------------------------
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum"
)

def synthesize_summary(transcriptions: list[str]) -> str:
    """Gera resumo em tópicos Markdown a partir de lista de transcrições"""
    full_text = " ".join(transcriptions)
    if not full_text.strip():
        return ""

    # limitar tamanho para não estourar a entrada do modelo
    full_text = full_text[-2000:]

    raw_summary = summarizer(
        full_text, 
        max_length=200, 
        min_length=50, 
        do_sample=False
    )[0]["summary_text"]

    # transformar em tópicos markdown
    topics = [f"- {sent.strip()}" for sent in raw_summary.split(". ") if sent.strip()]
    return "\n".join(topics)

# -------------------------------
# Utilidades
# -------------------------------
def get_md_filename() -> str:
    md_filename = input(
        "📄 Nome do arquivo Markdown (ex: transcricao.md). "
        "Deixe em branco para gerar automaticamente: "
    ).strip()
    if not md_filename:
        md_filename = datetime.now().strftime("transcricao-%Y-%m-%d-%Hh%Mm.md")
    elif not md_filename.endswith(".md"):
        md_filename += ".md"
    return os.path.join(TRANSCRIPTION_DIR, md_filename)

def select_input_devices() -> tuple[list[int], int]:
    print("\nDispositivos de entrada disponíveis:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{i}: {dev['name']} (Canais: {dev['max_input_channels']}, SR: {dev['default_samplerate']})")

    ids_str = input("\nDigite 1 ou 2 IDs de dispositivos separados por vírgula: ").strip()
    device_ids = [int(x) for x in ids_str.split(",")]
    if len(device_ids) == 0 or len(device_ids) > 2:
        raise ValueError("Você deve escolher 1 ou 2 dispositivos de entrada.")

    sample_rates = [int(devices[i]['default_samplerate']) for i in device_ids]
    sample_rate = min(sample_rates)

    print("\n✅ Dispositivos escolhidos:")
    for i, dev_id in enumerate(device_ids):
        print(f" - Canal {i} = {devices[dev_id]['name']} (SR: {sample_rate})")

    return device_ids, sample_rate

def record_chunk_multi(device_ids: list[int], sample_rate: int, duration: int) -> np.ndarray:
    audios = []
    for dev_id in device_ids:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=dev_id
        )
        audios.append(audio)
        sd.wait()
    if len(audios) == 1:
        return audios[0].flatten()
    else:
        return np.column_stack((audios[0].flatten(), audios[1].flatten()))

# -------------------------------
# Modelos
# -------------------------------
def load_models():
    print("🔍 Carregando modelo Whisper...")
    whisper_model = whisper.load_model(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Whisper carregado!")

    print("🔍 Carregando pipeline de diarização...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN
    )
    print("✅ Pipeline de diarização carregado!")

    return whisper_model, diarization_pipeline

# -------------------------------
# Alinhamento fala ↔️ speaker
# -------------------------------
def assign_speaker_to_segment(start: float, end: float, diarization) -> str:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= start and turn.end >= end:
            return speaker
    return "UNKNOWN"

# -------------------------------
# Loop Principal de Gravação
# -------------------------------
def live_transcription():
    md_path = get_md_filename()
    device_ids, sample_rate = select_input_devices()
    whisper_model, diarization_pipeline = load_models()

    transcribed_texts = []
    block_start_time = datetime.now()  # marca início do bloco de 20 minutos

    with open(md_path, "w", encoding="utf-8") as md_file:
        print("▶️ Iniciando transcrição em tempo real (Ctrl+C para parar)")
        try:
            while True:
                start_time = datetime.now()
                chunk = record_chunk_multi(device_ids, sample_rate, CHUNK_DURATION)
                temp_path = os.path.join(TEMP_DIR, "chunk.wav")
                write(temp_path, sample_rate, (chunk * 32767).astype(np.int16))

                diarization = diarization_pipeline(temp_path)
                result = whisper_model.transcribe(
                    temp_path, fp16=True, language=LANGUAGE, task="transcribe"
                )

                for seg in result["segments"]:
                    start = seg["start"]
                    end = seg["end"]
                    text = seg["text"].strip()
                    if not text:
                        continue

                    speaker = assign_speaker_to_segment(start, end, diarization)
                    channel_info = " [Estéreo]" if chunk.ndim == 2 else ""
                    timestamp = f"[{start_time.strftime('%H:%M:%S')} - {datetime.now().strftime('%H:%M:%S')}]"
                    line = f"{timestamp} [{start:.2f}-{end:.2f}] {speaker}{channel_info}: {text}"

                    print(line)
                    md_file.write(line + "\n")
                    md_file.flush()
                    transcribed_texts.append(text)

                # --- Verifica se já passou 20 minutos desde o último resumo ---
                if datetime.now() - block_start_time >= timedelta(minutes=20):
                    print("\n📝 Gerando síntese dos últimos 20 minutos...\n")
                    summary = synthesize_summary(transcribed_texts)
                    if summary:
                        md_file.write("\n\n## Resumo dos últimos 20 minutos\n")
                        md_file.write(summary + "\n\n")
                        md_file.flush()
                        print("✅ Resumo em tópicos gerado e salvo.")

                    # Reinicia o bloco
                    transcribed_texts = []
                    block_start_time = datetime.now()

        except KeyboardInterrupt:
            print("\n✅ Transcrição finalizada.")

            # Gera resumo final do que sobrou (mesmo se < 20 min)
            if transcribed_texts:
                print("\n📝 Gerando resumo final...\n")
                summary = synthesize_summary(transcribed_texts)
                if summary:
                    md_file.write("\n\n## Resumo final\n")
                    md_file.write(summary + "\n\n")
                    md_file.flush()
                    print("✅ Resumo final salvo.")

    shutil.rmtree(TEMP_DIR)
    print(f"✅ Arquivos temporários removidos. Transcrição salva em: {md_path}")

# -------------------------------
# Modo pós-processamento
# -------------------------------
def summarize_existing_file(path: str):
    if not os.path.exists(path):
        print(f"❌ Arquivo não encontrado: {path}")
        sys.exit(1)

    print(f"🔍 Lendo transcrição existente: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f.readlines() if line.strip()]

    print("📝 Gerando resumo...")
    summary = synthesize_summary(text_lines)

    if summary:
        summary_path = path.replace(".md", "-resumo.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Resumo da Transcrição\n\n")
            f.write(summary + "\n")
        print(f"✅ Resumo salvo em: {summary_path}")
    else:
        print("⚠️ Não foi possível gerar resumo (texto vazio).")

# -------------------------------
# Execução
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcrição e Resumo de Áudio")
    parser.add_argument("--summarize", type=str, help="Gerar resumo de um arquivo de transcrição existente")
    args = parser.parse_args()

    if args.summarize:
        summarize_existing_file(args.summarize)
    else:
        live_transcription()
