<p align="center">
  <img src="images/image-removebg-preview.png" alt="Aslan Logo" width="300">
</p>

<h1 align="center">🦁 Aslan (AI Service Language notes)</h1>

O **Aslan** é um ecossistema de inteligência artificial projetado para transformar reuniões, aulas e conversas em conhecimento estruturado e acionável. Integrado diretamente ao seu fluxo de trabalho e ao seu "Segundo Cérebro" (Obsidian), o Aslan automatiza desde a captura de áudio até a gestão de tarefas.

## 👥 A Irmandade da Nota (Agentes)

O projeto é composto por quatro agentes especializados:

### 🎤 Lúcia (The Listener)
A Lúcia é a porta de entrada do conhecimento. Ela vive acoplada à sua saída de áudio, capturando tudo o que acontece em suas reuniões (Teams, Zoom, Meet).
- **O que faz:** Captura áudio em tempo real, realiza transcrição de alta precisão, diarização (identifica quem está falando) e gera um resumo executivo automático.
- **Tecnologias:** OpenAI Whisper (via `faster-whisper`), PyAnnote.audio, e Ollama para sumarização local.
- **Saída:** Arquivos Markdown estruturados diretamente na sua pasta do Obsidian.

### 📚 Edmundo (The Archivist)
O Edmundo é o guardião da memória. Ele lê tudo o que a Lúcia transcreveu e está pronto para responder qualquer pergunta sobre o histórico de conversas.
- **O que faz:** Indexa as transcrições usando embeddings e atua como um sistema RAG (Retrieval-Augmented Generation). Você pode perguntar sobre decisões passadas, datas ou detalhes técnicos discutidos em reuniões de meses atrás.
- **Tecnologias:** `sentence-transformers` para busca semântica, Torch e Ollama (Mistral/Qwen).

### 🔗 Suzana (Preview)
A Suzana é a arquiteta de conexões. Ela analisa as notas geradas e busca relacioná-las com o que já existe no seu sistema de gestão.
- **O que faz:** Vincula as notas de reuniões a tarefas, projetos e conceitos existentes no seu Obsidian, criando uma teia de conhecimento interconectada.

### 🛠️ Pedro (Preview)
O Pedro é o agente de execução. Ele transforma palavras em ações concretas.
- **O que faz:** Com base nos relacionamentos gerados pela Suzana e nos pontos de ação identificados pela Lúcia, o Pedro cria automaticamente as tasks que precisam ser feitas, garantindo que nada se perca entre uma reunião e outra.

---

## 🛠️ Stack Técnica

O projeto preza pela **privacidade e processamento local**:
- **Linguagem:** Python 3.12+
- **Transcrição:** `faster-whisper` (Modelo Medium/Large)
- **Diarização:** `pyannote.audio` 3.1
- **LLM Local:** [Ollama](https://ollama.com/) (Modelos recomendados: `qwen2.5`, `mistral-nemo`)
- **Embeddings:** `all-mpnet-base-v2`
- **Interface de Áudio:** `sounddevice` / `pactl` (Linux/KDE integration)
- **Armazenamento:** Obsidian (Markdown)

---

## 🚀 Como Começar

### Pré-requisitos
1. **Ollama** instalado e rodando.
2. Modelos baixados: `ollama pull qwen2.5:7b-instruct-q5_K_M` e `ollama pull mistral-nemo`.
3. Um token do **HuggingFace** (para o PyAnnote).

### Instalação
1. Clone o repositório.
2. Crie um ambiente virtual: `python -m venv .venv`.
3. Instale as dependências: `pip install -r requirements.txt`.
4. Configure o arquivo `.env`:
   ```env
   OBSIDIAN_TRANSCRIPT_DIR="/caminho/para/seu/obsidian/transcricoes"
   HF_TOKEN="seu_token_aqui"
   ```

### Uso
- **Para rodar a Lúcia (Gravação/Transcrição):**
  ```bash
  python lucia.py
  ```
- **Para consultar o Edmundo (Busca/RAG):**
  ```bash
  python edmundo.py
  ```

---

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para mais detalhes.
