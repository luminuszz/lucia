#!/home/daviribeiro/projects/personal/lucia/venv/bin/python

import os
from datetime import datetime
import sounddevice as sd
import numpy as np
import torch
import whisper
from scipy.io.wavfile import write
import shutil
from pyannote.audio import Pipeline

# -------------------------------
# Configurações
# -------------------------------
CHUNK_DURATION = 20  # segundos por chunk
CHANNELS = 1  # mono
TRANSCRIPTION_DIR = "transcricoes"
TEMP_DIR = ".temp"
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------
# Nome do arquivo Markdown
# -------------------------------
md_filename = input(
    "📄 Digite o nome do arquivo Markdown para a transcrição (ex: transcricao.md). "
    "Se deixar em branco será gerado automaticamente: "
).strip()

if not md_filename:
    md_filename = datetime.now().strftime("transcricao-%Y-%m-%d-%Hh%Mm.md")
elif not md_filename.endswith(".md"):
    md_filename += ".md"

md_path = os.path.join(TRANSCRIPTION_DIR, md_filename)

# -------------------------------
# Seleção do dispositivo de entrada
# -------------------------------
print("\nDispositivos de entrada disponíveis:\n")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']} (Canais: {dev['max_input_channels']}, SR: {dev['default_samplerate']})")

device_id = int(input("\nDigite o ID do dispositivo de entrada: "))
device_info = sd.query_devices(device_id)
SAMPLE_RATE = int(device_info['default_samplerate'])
CHANNELS = min(device_info['max_input_channels'], CHANNELS)

print(f"\nUsando dispositivo: {device_info['name']} (SR: {SAMPLE_RATE}, Canais: {CHANNELS})")

# -------------------------------
# Carregar modelo Whisper
# -------------------------------
print("🔍 Carregando modelo Whisper (medium) na GPU...")
model = whisper.load_model("medium", device="cuda")
print("✅ Modelo carregado!")

# -------------------------------
# Carregar pipeline de diarização
# -------------------------------
print("🔍 Carregando pipeline de diarização de vozes...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
print("✅ Pipeline de diarização carregado!")

# -------------------------------
# Função para gravar áudio
# -------------------------------
def record_chunk(duration):
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=device_id
    )
    sd.wait()
    return audio.flatten()

# -------------------------------
# Loop principal de transcrição
# -------------------------------
with open(md_path, "w", encoding="utf-8") as md_file:
    print("▶️ Iniciando transcrição em tempo real (Ctrl+C para parar)")
    try:
        while True:
            start_time = datetime.now()
            chunk = record_chunk(CHUNK_DURATION)

            # Salvar temporário
            temp_path = os.path.join(TEMP_DIR, "chunk.wav")
            write(temp_path, SAMPLE_RATE, (chunk * 32767).astype(np.int16))

            # Diarização
            diarization = pipeline(temp_path)

            # Transcrição
            result = model.transcribe(temp_path, fp16=True, language="pt", task="transcribe")
            text = result["text"].strip()

            if text:
                end_time = datetime.now()
                timestamp = f"[{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}]"
                
                # Combina diarização e transcrição
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_sec = int(turn.start)
                    end_sec = int(turn.end)
                    output_line = f"[{start_sec}-{end_sec}] {speaker}: {text}"
                    print(output_line)
                    md_file.write(output_line + "\n")
                md_file.flush()

    except KeyboardInterrupt:
        print("\n✅ Transcrição finalizada.")

# Limpar arquivos temporários
shutil.rmtree(TEMP_DIR)
print(f"✅ Arquivos temporários removidos. Transcrição salva em: {md_path}")
