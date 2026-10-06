"""
Configuration centrale du pipeline RAG-PMBOK

Un seul endroit pour tous les paramètres du projet Every session ajoutera
ses propres réglages ici (embeddings, FAISS, LLM...), plutôt que d'éparpiller
des constantes dans chaque script
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charge automatiquement les variables du fichier .env à la racine du projet
# (OPENROUTER_API_KEY pour la vision, GROQ_API_KEY pour la génération finale)si
# le fichier n'existe pas encore, ne fait rien (pas d'erreur), les clés
# resteront simplement absentes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Chemins ---
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_OUTPUT_PATH = DATA_PROCESSED_DIR / "chunks.json"

# --- Session 1 : Extraction & Chunking ---
CHUNK_TARGET_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50
HEADING_FONT_DIFF_THRESHOLD = 1.5
REPEATED_LINE_MIN_COUNT = 3

# Camelot (extraction de tableaux) est lent sur un gros PDF on découpe le
# traitement en lots de pages plutôt qu'une seule passe sur tout le document.
TABLE_EXTRACTION_BATCH_SIZE = 100

# Bascule simple pour activer/désactiver l'étape de détection des formules
# (utile pour la mettre de côté temporairement sans supprimer le code)
ENABLE_FORMULAS_EXTRACTION = True

# --- Session 2 : Embeddings & FAISS (à venir) ---
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = DATA_PROCESSED_DIR / "pmbok.index"

# --- Session 3 : Retrieval & Reranking (à venir) ---
TOP_K_RETRIEVAL = 5
CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# --- Session 4 : Génération ---
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
TOP_N_CONTEXT = 4
MAX_ANSWER_TOKENS = 900
MAX_CONTEXT_TOKENS = 4500

# --- Description des figures par LLM à vision (session 1, étape 4bis) ---
OPENROUTER_VISION_MODEL_NAME = "google/gemini-2.5-flash-lite"

# --- Clés API, chargées depuis le fichier .env (voir .env.example) ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

RAGAS_JUDGE_MODEL = "openai/gpt-oss-20b"