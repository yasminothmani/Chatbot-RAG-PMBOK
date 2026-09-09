"""
Reranking par CrossEncoder — deuxième moitié du Retriever.

Contrairement à FAISS (qui encode la question et chaque chunk séparément,
puis compare leurs vecteurs déjà calculés), le CrossEncoder lit la question
ET le chunk ENSEMBLE, en une seule passe  ce qui lui permet de juger la
pertinence de façon beaucoup plus fine, en tenant compte de l'interaction
réelle entre les deux textes.

Ce mode de fonctionnement est plus lent, donc appliqué uniquement aux k
candidats déjà retrouvés par FAISS (pas à tout le corpus).

Modèle retenu : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 — version
multilingue, cohérente avec le choix fait pour l'embedding (le PMBOK est en
anglais, les questions peuvent être posées en français).
"""

import config


def load_reranker(model_name: str = None):
    from sentence_transformers import CrossEncoder
    model_name = model_name or config.CROSS_ENCODER_MODEL_NAME
    return CrossEncoder(model_name)


def rerank(question: str, candidates: list[dict], model=None) -> list[dict]:
    """Réordonne une liste de chunks candidats selon leur pertinence réelle
    par rapport à la question, jugée par le CrossEncoder. Chaque candidat
    reçoit un score `rerank_score` ; la liste est retournée triée du plus
    au moins pertinent."""
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
    print(f"Question : {question}\n")

    candidates = retrieve(question)

    print("=" * 60)
    print("AVANT reranking (ordre FAISS)")
    print("=" * 60)
    for i, c in enumerate(candidates):
        print(f"#{i + 1} — {c['chunk_id']} (distance FAISS : {c['faiss_distance']:.2f})")

    reranked = rerank(question, candidates)

    print(f"\n{'=' * 60}")
    print("APRÈS reranking (ordre CrossEncoder)")
    print("=" * 60)
    for i, c in enumerate(reranked):
        print(f"#{i + 1} — {c['chunk_id']} (score CrossEncoder : {c['rerank_score']:.3f})")
        print(c["text"][:200])
        print()