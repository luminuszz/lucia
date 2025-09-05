import os
import shutil
import sounddevice as sd
import numpy as np
import torch
import whisper
from datetime import datetime
from scipy.io.wavfile import write

# Configurações
CHUNK_DURATION = 5  # duração de cada chunk em segundos

# Criar pastas
temp_dir = ".temp"
transcription_dir = "transcricoes"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(transcription_dir, exist_ok=True)

# Input para nome do arquivo
md_filename = input("📄 Digite o nome do arquivo Markdown para a transcrição (ex: transcricao.md). "
                    "Se deixar em branco será gerado automaticamente: ").strip()

if not md_filename:
    # Nome automático baseado na data e hora
    md_filename = datetime.now().strftime("transcricao-%Y-%m-%d-%Hh%Mm.md")
elif not md_filename.endswith(".md"):
    md_filename += ".md"

# Caminho final dentro da pasta "transcricoes"
md_path = os.path.join(transcription_dir, md_filename)

# Listar dispositivos de entrada
print("\nDispositivos de entrada disponíveis:\n")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']} (Canais de entrada: {dev['max_input_channels']}, Default SR: {dev['default_samplerate']})")

device_id = int(input("\nDigite o ID do dispositivo de entrada (output do fone): "))
device_info = sd.query_devices(device_id)
SAMPLE_RATE = int(device_info['default_samplerate'])
CHANNELS = min(device_info['max_input_channels'], 1)  # mono

print(f"\nUsando dispositivo: {device_info['name']} (SR: {SAMPLE_RATE}, Canais: {CHANNELS})")

# Carregar modelo Whisper
print("🔍 Carregando modelo Whisper (large) na GPU...")
model = whisper.load_model("large", device="cuda")
print("✅ Modelo carregado!")

# Função para gravar chunk
def record_chunk():
    audio = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype='float32',
                   device=device_id)
    sd.wait()
    return audio.flatten()

# Abrir arquivo Markdown para salvar transcrição
with open(md_path, "w", encoding="utf-8") as md_file:
    print("▶️ Iniciando transcrição em tempo real (Ctrl+C para parar)")
    try:
        while True:
            # Gravar áudio
            chunk = record_chunk()

            # Salvar temporariamente
            temp_path = os.path.join(temp_dir, "chunk.wav")
            write(temp_path, SAMPLE_RATE, (chunk * 32767).astype(np.int16))

            # Carregar áudio para Whisper
            result = model.transcribe(temp_path, fp16=False)
            text = result["text"].strip()
            if text:
                print(text)
                md_file.write(text + "\n")
                md_file.flush()  # força salvar no disco a cada chunk

    except KeyboardInterrupt:
        print("\n✅ Transcrição finalizada.")

# Limpar arquivos temporários
shutil.rmtree(temp_dir)
print(f"✅ Arquivos temporários removidos. Transcrição salva em: {md_path}")
