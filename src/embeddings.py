"""
Génère les embeddings (vecteurs numériques) de chaque chunk.

Modèle utilisé : paraphrase-multilingual-MiniLM-L12-v2 — variante multilingue
de MiniLM, retenue plutôt que all-MiniLM-L6-v2 (celui du rapport initial) pour
une raison précise : le corpus PMBOK est en anglais, mais les descriptions de
figures générées en session 1 sont en français, et les utilisateurs du chatbot
poseront probablement leurs questions en français aussi. Un modèle purement
anglais donnerait un mauvais retrieval sur tout ce qui n'est pas en anglais.

Sortie : un vecteur de 384 dimensions par chunk, dans le même ordre que la
liste de chunks fournie en entrée — l'ordre est important, c'est lui qui relie
chaque vecteur à son chunk d'origine au moment de l'indexation FAISS.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def load_chunks() -> list[dict]:
    """Recharge les chunks générés en session 1 (texte, tableaux, figures, formules)."""
    with open(config.CHUNKS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], model_name: str = None) -> np.ndarray:
    """Encode chaque chunk['text'] en vecteur. Retourne un tableau numpy de
    forme (nombre_de_chunks, 384) — une ligne par chunk, dans le même ordre
    que la liste fournie."""
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Encodage de {len(texts)} chunks avec {model_name}...")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # FAISS attend des vecteurs en float32, SentenceTransformers retourne
    # parfois du float64 selon la configuration — on force le bon type.
    return vectors.astype("float32")


if __name__ == "__main__":
    chunks = load_chunks()
    vectors = embed_chunks(chunks)
    print(f"\n{vectors.shape[0]} vecteurs générés, dimension {vectors.shape[1]}")
    print(f"Exemple — chunk 0 ({chunks[0]['chunk_id']}), 5 premières valeurs du vecteur :")
    print(vectors[0][:5])
