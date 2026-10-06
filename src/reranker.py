"""
Reranking par CrossEncoder — deuxième moitié du Retriever.

Accepte maintenant un modèle déjà chargé (`model`) pour éviter de recharger
le CrossEncoder à chaque appel — indispensable pour Streamlit.
"""

import config


def load_reranker(model_name: str = None):
    from sentence_transformers import CrossEncoder
    model_name = model_name or config.CROSS_ENCODER_MODEL_NAME
    return CrossEncoder(model_name)


def rerank(question: str, candidates: list[dict], model=None) -> list[dict]:
    """Réordonne une liste de chunks candidats selon leur pertinence réelle,
    jugée par le CrossEncoder. Si `model` est fourni (déjà chargé), il est
    réutilisé ; sinon un nouveau est chargé à la volée."""
    model = model or load_reranker()

    pairs = [(question, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


if __name__ == "__main__":
    import sys
    from retrieval import retrieve

    question = sys.argv[1] if len(sys.argv) > 1 else "Comment gérer l'engagement des parties prenantes ?"
    candidates = retrieve(question)
    reranked = rerank(question, candidates)
    for i, c in enumerate(reranked):
        print(f"#{i + 1} — {c['chunk_id']} (score : {c['rerank_score']:.3f})")