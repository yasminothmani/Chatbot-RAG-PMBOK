"""
Interroge FAISS avec une vraie question utilisateur — contrairement au test
de la session 2, qui réutilisait un chunk déjà existant comme requête
(pratique pour tester rapidement, mais pas représentatif de l'usage réel).

C'est la première moitié du Retriever : transformer la question en vecteur,
puis chercher les k chunks les plus proches dans l'index FAISS déjà construit.
La deuxième moitié (reranking) se trouve dans reranker.py.
"""

import numpy as np

import config
from vector_store import load_index, search
from embeddings import load_chunks


def embed_query(question: str, model_name: str = None) -> np.ndarray:
    """Encode la question avec le MÊME modèle que celui utilisé pour les
    chunks — indispensable pour que question et chunks soient comparables
    dans le même espace vectoriel."""
    from sentence_transformers import SentenceTransformer
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    model = SentenceTransformer(model_name)
    vector = model.encode([question], convert_to_numpy=True)[0]
    return vector.astype("float32")


def retrieve(question: str, k: int = None) -> list[dict]:
    """Pipeline complet de recherche : question -> vecteur -> FAISS ->
    liste de chunks candidats, chacun enrichi de sa distance FAISS."""
    k = k or config.TOP_K_RETRIEVAL
    index = load_index()
    chunks = load_chunks()

    query_vector = embed_query(question)
    distances, indices = search(index, query_vector, k=k)

    results = []
    for dist, idx in zip(distances, indices):
        candidate = dict(chunks[idx])
        candidate["faiss_distance"] = float(dist)
        results.append(candidate)
    return results


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "Comment gérer l'engagement des parties prenantes ?"
    print(f"Question : {question}\n")

    results = retrieve(question)
    print(f"{len(results)} candidats retrouvés par FAISS (avant reranking)\n")
    for i, r in enumerate(results):
        print(f"#{i + 1} — {r['chunk_id']} (distance FAISS : {r['faiss_distance']:.2f})")
        print(r["text"][:200])
        print()