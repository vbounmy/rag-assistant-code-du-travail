import os
from pathlib import Path

# Projet
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT_DIR / "prompts"

# Chemins de fichiers
ENV_PATH = ROOT_DIR / ".env"
CHROMA_PERSIST_PATH = ROOT_DIR / "chroma_db"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_system.txt"
MODERATOR_PROMPT_PATH = PROMPTS_DIR / "moderator_system.txt"

# Données
DATA_DIR = ROOT_DIR / "data"

# Modèles
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
MODERATOR_MODEL = "safeguard-0.1.0"

# Chargement de l'environnement
if ENV_PATH.exists():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")