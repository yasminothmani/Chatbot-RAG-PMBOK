"""
Évaluation qualitative en lot — session 6.

Lance toutes les questions de test d'un coup, avec les modèles chargés une
seule fois, et écrit un rapport markdown (mode de routage, sources, réponse)
dans data/processed/evaluation_results.md. Une commande, un fichier à relire.

Usage :
    python evaluate.py                  # les 10 questions par défaut
    python evaluate.py "Ma question ?"  # une seule question
"""

import sys
import time
from datetime import datetime

import config
from embeddings import load_chunks
from vector_store import load_index
from retrieval import smart_retrieve_v3, BROAD_TOP_N
from reranker import rerank, load_reranker
from generation import generate_answer, detect_language

QUESTIONS = [
    ("Q1 · ciblée",   "Qu'est-ce que l'approche adaptative en gestion de projet ?"),
    ("Q2 · type",     "Quels sont les outils de gestion de projet mentionnés dans le PMBOK ?"),
    ("Q3 · domaine",  "What are the 8 performance domains?"),
    ("Q4 · broad",    "Combien de méthodes d'estimation existe-t-il ?"),
    ("Q5 · formule",  "Comment calcule-t-on le CPI et le SPI ?"),
    ("Q6 · figure",   "Que montre la figure 2-3 sur l'engagement des parties prenantes ?"),
    ("Q7 · tableau",  "Quelle est la différence entre un projet, un programme et un portefeuille ?"),
    ("Q8 · hors sujet", "Quelle est la capitale de la France ?"),
    ("Q9 · limite",   "Que dit le PMBOK sur la méthode Scrum en détail ?"),
    ("Q10 · anglais", "How do I calculate earned value?"),
]

OUTPUT_PATH = config.DATA_PROCESSED_DIR / "evaluation_results.md"


def answer(question, embed_model, cross_model, groq_client, index, chunks):
    candidates, mode = smart_retrieve_v3(question, embed_model=embed_model, index=index, chunks=chunks)
    if mode in ("type", "domaine"):
        result = generate_answer(question, candidates, top_n=len(candidates), client=groq_client)
    elif mode == "formule":
        formulas = [c for c in candidates if c.get("content_type") == "formule"]
        others = rerank(question, [c for c in candidates if c.get("content_type") != "formule"], model=cross_model)
        result = generate_answer(question, formulas + others, top_n=len(formulas) + config.TOP_N_CONTEXT, client=groq_client)
    elif mode == "broad":
        result = generate_answer(question, rerank(question, candidates, model=cross_model), top_n=BROAD_TOP_N, client=groq_client)
    else:
        result = generate_answer(question, rerank(question, candidates, model=cross_model), client=groq_client)
    result["mode"] = mode
    return result


def main():
    if len(sys.argv) > 1:
        questions = [("Q · libre", " ".join(sys.argv[1:]))]
    else:
        questions = QUESTIONS

    print("Chargement des modèles (une seule fois)...")
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    cross_model = load_reranker()
    groq_client = Groq(api_key=config.GROQ_API_KEY)
    index = load_index()
    chunks = load_chunks()

    lines = [f"# Évaluation qualitative — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for label, q in questions:
        print(f"\n{label} : {q}")
        t0 = time.time()
        try:
            r = answer(q, embed_model, cross_model, groq_client, index, chunks)
            elapsed = time.time() - t0
            print(f"   mode={r['mode']}  langue={detect_language(q)}  sources={len(r['sources'])}  {elapsed:.1f}s")
            lines += [
                f"## {label}", f"**Question :** {q}", "",
                f"**Mode :** {r['mode']} · **Langue détectée :** {detect_language(q)} · **Durée :** {elapsed:.1f}s",
                f"**Sources :** {', '.join(r['sources'])}", "",
                r["answer"], "", "---", "",
            ]
        except Exception as e:
            print(f"   ERREUR : {e}")
            lines += [f"## {label}", f"**Question :** {q}", "", f"**ERREUR :** {e}", "", "---", ""]
        # Palier gratuit Groq : 8 000 tokens/minute — on espace les appels
        time.sleep(8)

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport écrit : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
