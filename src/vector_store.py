"""
Construit et interroge l'index FAISS — la base vectorielle du projet.

Choix retenu (cohérent avec le rapport, section 3.2.3) : IndexFlatL2, une
recherche exacte par distance euclidienne. Pas d'approximation (contrairement
à IndexIVFPQ) — adapté à un corpus de 441 chunks, largement en dessous du
seuil où l'approximation deviendrait nécessaire pour rester rapide.

L'index est sauvegardé sur disque après construction, pour ne pas avoir à
ré-encoder les 441 chunks à chaque lancement du chatbot.
"""

import json
import numpy as np
import faiss

import config


def build_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """Crée un index FAISS et y insère tous les vecteurs. L'ordre d'insertion
    est conservé — le vecteur à la position i dans `vectors` correspond au
    chunk à la position i dans la liste de chunks d'origine."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


def save_index(index: faiss.IndexFlatL2, path: str = None):
    path = path or str(config.FAISS_INDEX_PATH)
    config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)
    print(f"Index sauvegardé dans {path}")


def load_index(path: str = None) -> faiss.IndexFlatL2:
    path = path or str(config.FAISS_INDEX_PATH)
    return faiss.read_index(path)


def search(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int = 5):
    """Cherche les k chunks les plus proches d'un vecteur de requête.
    Retourne (distances, indices) — indices pointe vers la position du
    chunk dans la liste d'origine (chunks.json), pas vers son contenu
    directement : il faut recharger chunks.json pour retrouver le texte."""
    query_vector = query_vector.reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


if __name__ == "__main__":
    from embeddings import load_chunks, embed_chunks

    chunks = load_chunks()
    vectors = embed_chunks(chunks)

    print("\nConstruction de l'index...")
    index = build_index(vectors)
    print(f"Index construit : {index.ntotal} vecteurs, dimension {index.d}")

    save_index(index)

    print("\nTest de recherche (vecteur de requête = chunk 10 lui-même)...")
    distances, indices = search(index, vectors[10], k=5)
    print(f"Indices trouvés : {indices}")
    print(f"Distances : {distances}")
    for idx in indices:
        print(f"  - {chunks[idx]['chunk_id']} : {chunks[idx]['text'][:80]}...")