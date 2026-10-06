"""
Évaluation quantitative avec RAGAS — session 6.

evaluate.py produit un rapport à relire à la main. Ce script mesure la même
chose avec des métriques chiffrées, reconnues dans la littérature RAG :

    faithfulness      La réponse est-elle entièrement soutenue par les extraits
                      fournis ? (0 = inventée, 1 = chaque affirmation est
                      dans le contexte). C'est LA métrique anti-hallucination.
    answer_relevancy  La réponse répond-elle vraiment à la question posée,
                      sans digression ni hors-sujet ?
    context_precision Les extraits retrouvés en tête sont-ils les pertinents ?
                      (qualité du classement du retriever + reranker)
    context_recall    Les extraits retrouvés contiennent-ils tout ce qu'il
                      faut pour produire la réponse de référence ?
                      (couverture du retriever)

Les deux premières jugent la GÉNÉRATION, les deux dernières le RETRIEVAL.

RAGAS a besoin d'un LLM « juge » pour décomposer les réponses en
affirmations et les vérifier. On utilise Groq via LangChain, car RAGAS est
construit sur l'interface LangChain (c'est la seule raison de cette
dépendance, notre pipeline lui-même ne l'utilise pas). Le juge est par
défaut un modèle DIFFÉRENT de celui qui génère les réponses, pour que le
modèle ne se note pas lui-même ; on peut le changer avec RAGAS_JUDGE_MODEL
dans config.py. Le rapport indique toujours quel juge a été utilisé, et
signale explicitement si juge et générateur sont identiques.

Usage :
    python evaluate_ragas.py
Sortie :
    data/processed/ragas_results.md
"""

import time
from datetime import datetime

from langchain_groq import ChatGroq
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

import config
from embeddings import load_chunks
from vector_store import load_index
from reranker import load_reranker
from evaluate import answer

# Juge : gpt-oss-20b, différent du générateur (gpt-oss-120b). Les modèles
# gpt-oss « réfléchissent » avant de répondre et cette réflexion consomme
# les tokens de sortie ; sans plafond élevé et sans effort de raisonnement
# réduit, la réponse JSON attendue par RAGAS est coupée
# (LLMDidNotFinishException), d'où les deux réglages ci-dessous.
JUDGE_MODEL = getattr(config, "RAGAS_JUDGE_MODEL", "openai/gpt-oss-20b")
JUDGE_KWARGS = {"reasoning_effort": "low"} if "gpt-oss" in JUDGE_MODEL else {}
SAME_MODEL = JUDGE_MODEL == config.GROQ_MODEL_NAME
OUTPUT_PATH = config.DATA_PROCESSED_DIR / "ragas_results.md"
PAUSE_BETWEEN_QUESTIONS = 8

# Les questions hors périmètre (capitale de la France, Scrum) sont
# volontairement absentes : RAGAS mesure la qualité d'une réponse par rapport
# au contexte et à une référence, ce qui n'a pas de sens pour un refus.
# Chaque question a une réponse de référence (ground truth) écrite à partir
# du PMBOK 7, nécessaire pour context_recall.
DATASET = [
    {
        "question": "Qu'est-ce que l'approche adaptative en gestion de projet ?",
        "reference": "L'approche adaptative est une approche de développement itérative et incrémentale. "
                     "Le travail est découpé en itérations qui produisent chacune un livrable fonctionnel, "
                     "le backlog est re-priorisé à partir des retours obtenus, et l'approche s'appuie sur "
                     "des méthodes agiles et la planification basée sur le flux. Elle convient quand les "
                     "exigences sont incertaines ou évoluent souvent.",
    },
    {
        "question": "Quels sont les outils de gestion de projet mentionnés dans le PMBOK ?",
        "reference": "Le PMBOK 7 mentionne des outils de communication (réunions, courriels, messagerie, "
                     "intranet, réseaux sociaux), des tableaux de bord qui agrègent des métriques sous forme "
                     "de graphiques, des radiateurs d'information (grands affichages visibles), ainsi que des "
                     "modèles, méthodes et artefacts. Le choix des outils et logiciels relève de l'adaptation "
                     "(tailoring) et aucun produit spécifique n'est prescrit.",
    },
    {
        "question": "What are the 8 performance domains?",
        "reference": "The eight performance domains are: Stakeholder, Team, Development Approach and Life "
                     "Cycle, Planning, Project Work, Delivery, Measurement, and Uncertainty.",
    },
    {
        "question": "Combien de méthodes d'estimation existe-t-il ?",
        "reference": "Le PMBOK 7 répertorie neuf méthodes d'estimation : le groupement par affinité, "
                     "l'estimation par analogie, les points de fonction, l'estimation multipoint, "
                     "l'estimation paramétrique, l'estimation relative, l'estimation à point unique, "
                     "l'estimation en points d'histoire et le Wideband Delphi.",
    },
    {
        "question": "Comment calcule-t-on le CPI et le SPI ?",
        "reference": "Le CPI (indice de performance des coûts) se calcule par CPI = EV / AC, la valeur acquise "
                     "divisée par le coût réel. Le SPI (indice de performance des délais) se calcule par "
                     "SPI = EV / PV, la valeur acquise divisée par la valeur planifiée.",
    },
    {
        "question": "Que montre la figure 2-3 sur l'engagement des parties prenantes ?",
        "reference": "La figure 2-3 présente l'engagement efficace des parties prenantes comme un processus "
                     "continu et itératif composé des étapes identifier, comprendre, analyser, prioriser, "
                     "engager et suivre (monitor), répété tout au long du projet.",
    },
    {
        "question": "Quelle est la différence entre un projet, un programme et un portefeuille ?",
        "reference": "Un projet est une entreprise temporaire visant à créer un produit, service ou résultat "
                     "unique. Un programme est un ensemble de projets, sous-programmes et activités liés, "
                     "gérés de façon coordonnée pour obtenir des bénéfices impossibles à obtenir séparément. "
                     "Un portefeuille regroupe des projets, programmes et opérations gérés collectivement "
                     "pour atteindre des objectifs stratégiques.",
    },
    {
        "question": "How do I calculate earned value?",
        "reference": "Earned value (EV) is the measure of work performed expressed in terms of the budget "
                     "authorized for that work. It is used in the earned value formulas: CV = EV - AC, "
                     "SV = EV - PV, CPI = EV / AC and SPI = EV / PV.",
    },
]


class SentenceTransformerEmbeddings(BaseRagasEmbeddings):
    """Adaptateur : RAGAS attend un objet avec embed_query / embed_documents
    (et leurs versions async). On réutilise le modèle d'embedding déjà chargé
    pour le retrieval, le même que celui de l'index FAISS."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def build_samples():
    print("Chargement des modèles (une seule fois)...")
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    cross_model = load_reranker()
    groq_client = Groq(api_key=config.GROQ_API_KEY)
    index = load_index()
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}

    samples, meta = [], []
    for i, item in enumerate(DATASET, 1):
        print(f"\n[{i}/{len(DATASET)}] {item['question']}")
        start = time.time()
        result = answer(item["question"], embed_model, cross_model, groq_client, index, chunks)
        contexts = [by_id[cid]["text"] for cid in result["sources"] if cid in by_id]
        print(f"   mode={result['mode']}  extraits={len(contexts)}  {time.time() - start:.1f}s")
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=item["reference"],
        ))
        meta.append({"mode": result["mode"], "n_contexts": len(contexts)})
        if i < len(DATASET):
            time.sleep(PAUSE_BETWEEN_QUESTIONS)
    return samples, meta, embed_model


def run_ragas(samples, embed_model):
    print(f"\nÉvaluation RAGAS avec le juge {JUDGE_MODEL}" + (" (identique au générateur !)" if SAME_MODEL else "") + "...")
    judge = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0, max_tokens=8000,
                                         model_kwargs=JUDGE_KWARGS, groq_api_key=config.GROQ_API_KEY))
    embeddings = SentenceTransformerEmbeddings(embed_model)
    # Un seul appel à la fois et des attentes longues : le palier gratuit de
    # Groq limite les tokens par minute, et RAGAS fait plusieurs appels par
    # question et par métrique.
    run_config = RunConfig(max_workers=1, timeout=180, max_retries=8, max_wait=60)
    return evaluate(
        EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )


def write_report(result, samples, meta):
    df = result.to_pandas()
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    lines = [f"# Évaluation RAGAS — {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines.append(f"Générateur : `{config.GROQ_MODEL_NAME}` · Juge : `{JUDGE_MODEL}` · "
                 f"Embeddings : `{config.EMBEDDING_MODEL_NAME}` · {len(samples)} questions")
    if SAME_MODEL:
        lines += ["", "> Attention : le juge est le même modèle que le générateur. Les scores restent "
                  "informatifs (le juge vérifie des affirmations contre des extraits, sans savoir qu'il "
                  "s'agit de sa propre réponse), mais un juge distinct est préférable."]
    lines += ["", "## Scores moyens", "", "| Métrique | Score | Ce que ça mesure |", "|---|---|---|"]
    descriptions = {
        "faithfulness": "la réponse ne contient que des affirmations présentes dans les extraits (anti-hallucination)",
        "answer_relevancy": "la réponse répond à la question posée, sans hors-sujet",
        "context_precision": "les extraits pertinents sont classés en tête (retriever + reranker)",
        "context_recall": "les extraits couvrent tout ce qu'il faut pour la réponse de référence (retriever)",
    }
    for c in cols:
        lines.append(f"| {c} | **{df[c].mean():.3f}** | {descriptions[c]} |")

    lines += ["", "## Détail par question", "",
              "| # | Question | Mode | Extraits | faithfulness | answer_relevancy | context_precision | context_recall |",
              "|---|---|---|---|---|---|---|---|"]
    for i, (s, m) in enumerate(zip(samples, meta), 1):
        row = df.iloc[i - 1]
        vals = " | ".join(f"{row[c]:.2f}" if row[c] == row[c] else "—" for c in cols)
        lines.append(f"| {i} | {s.user_input} | {m['mode']} | {m['n_contexts']} | {vals} |")

    lines += ["", "## Réponses générées", ""]
    for i, s in enumerate(samples, 1):
        lines += [f"### {i}. {s.user_input}", "", s.response.strip(), "",
                  f"*Référence : {s.reference}*", "", "---", ""]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    samples, meta, embed_model = build_samples()
    result = run_ragas(samples, embed_model)
    write_report(result, samples, meta)
    print("\nScores moyens :")
    df = result.to_pandas()
    for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"   {c:<20} {df[c].mean():.3f}")
    print(f"\nRapport écrit : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()