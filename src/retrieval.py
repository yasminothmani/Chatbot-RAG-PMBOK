"""
Interroge FAISS avec une vraie question utilisateur, et route automatiquement
les questions d'énumération vers un filtrage par métadonnées plutôt que la
recherche par similarité classique.

Toutes les fonctions acceptent maintenant des ressources déjà chargées
(embed_model, index, chunks) en paramètres optionnels — indispensable pour
Streamlit, où l'on charge les modèles UNE SEULE FOIS au démarrage de
l'application (via st.cache_resource) plutôt qu'à chaque question posée.
Sans ces paramètres, le comportement reste identique à avant (chaque appel
recharge son propre modèle) — donc les scripts en ligne de commande déjà
utilisés continuent de fonctionner sans rien changer.
"""

import numpy as np

import config
from vector_store import load_index, search
from embeddings import load_chunks


def embed_query(question: str, model_name: str = None, model=None) -> np.ndarray:
    """Encode la question avec le MÊME modèle que celui utilisé pour les
    chunks. Si `model` est fourni (déjà chargé), on le réutilise ; sinon
    on en charge un nouveau à la volée (mode ligne de commande)."""
    if model is None:
        from sentence_transformers import SentenceTransformer
        model_name = model_name or config.EMBEDDING_MODEL_NAME
        model = SentenceTransformer(model_name)
    vector = model.encode([question], convert_to_numpy=True)[0]
    return vector.astype("float32")


# --- Recherche hybride : dense (FAISS) + lexicale (BM25), fusion RRF ---

import re as _re

_bm25_cache = {}


def _tokenize(text: str) -> list[str]:
    return [t for t in _re.findall(r"[a-zà-ÿ0-9][a-zà-ÿ0-9\-]*", text.lower()) if len(t) > 1]


def get_bm25(chunks: list[dict]):
    """Index BM25 sur les textes des chunks, construit une fois et mis en cache
    (clé : nombre de chunks — le corpus ne change pas en cours d'exécution)."""
    from rank_bm25 import BM25Okapi
    key = len(chunks)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    return _bm25_cache[key]


def retrieve(question: str, k: int = None, embed_model=None, index=None, chunks=None,
             exclude_types: tuple = ()) -> list[dict]:
    """Recherche hybride. Retourne les k meilleurs chunks (bruit exclu), chacun
    avec faiss_distance (si classé par FAISS) et rrf_score.

    `exclude_types` écarte certains content_type du résultat (ex. "figure") :
    pour une question de liste, les descriptions de figures — nombreuses et
    pleines de vocabulaire PMBOK — peuvent occuper tous les premiers rangs et
    chasser le texte de la section qui contient réellement la réponse."""
    k = k or config.TOP_K_RETRIEVAL
    index = index if index is not None else load_index()
    chunks = chunks if chunks is not None else load_chunks()
    pool = k * 3  # candidats par méthode avant fusion

    # 1. Dense
    query_vector = embed_query(question, model=embed_model)
    distances, indices = search(index, query_vector, k=pool)
    dense_rank = {}
    for rank, (dist, idx) in enumerate(zip(distances, indices)):
        if idx >= 0:
            dense_rank[int(idx)] = (rank, float(dist))

    # 2. Lexical
    bm25 = get_bm25(chunks)
    scores = bm25.get_scores(_tokenize(question))
    lexical_order = [int(i) for i in np.argsort(scores)[::-1][:pool] if scores[i] > 0]
    lexical_rank = {idx: rank for rank, idx in enumerate(lexical_order)}

    # 3. Fusion RRF : score = somme de 1/(60 + rang) sur les méthodes où le chunk apparaît
    fused = {}
    for idx, (rank, _) in dense_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
    for idx, rank in lexical_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)

    results = []
    for idx in sorted(fused, key=lambda i: fused[i], reverse=True):
        if is_noise_chunk(chunks[idx]) or chunks[idx].get("content_type") in exclude_types:
            continue
        candidate = dict(chunks[idx])
        candidate["rrf_score"] = fused[idx]
        if idx in dense_rank:
            candidate["faiss_distance"] = dense_rank[idx][1]
        results.append(candidate)
        if len(results) == k:
            break
    return results


def add_following_chunks(candidates: list[dict], chunks: list[dict], top: int = 5, span: int = 2,
                         exclude_types: tuple = ()) -> list[dict]:
    """Ajoute, pour les `top` meilleurs candidats, les `span` chunks qui les
    SUIVENT dans le document.

    Pourquoi : le chunking coupe tous les ~2000 caractères, sans égard pour la
    structure. Une liste ou une section qui dépasse un chunk se retrouve donc
    répartie sur deux ou trois chunks consécutifs, souvent avec des étiquettes
    différentes (les principes 1-7 sont dans un chunk "principe", les
    principes 8-12 dans les deux suivants, étiquetés "processus"). La
    recherche trouve le premier ; la suite est juste derrière lui.

    On ne décide pas ici si les voisins sont utiles : ils rejoignent le lot de
    candidats, et c'est le reranker (CrossEncoder) qui juge ensuite de leur
    pertinence par rapport à la question. Les voisins hors sujet tombent."""
    position = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    present = {c["chunk_id"] for c in candidates}
    extra = []
    for c in candidates[:top]:
        i = position.get(c["chunk_id"])
        if i is None or is_noise_chunk(c):
            continue   # un sommaire ou un index n'a pas de "suite" utile
        for j in range(i + 1, min(i + 1 + span, len(chunks))):
            neighbor = chunks[j]
            if (neighbor["chunk_id"] in present or is_noise_chunk(neighbor)
                    or neighbor.get("content_type") in exclude_types):
                continue
            present.add(neighbor["chunk_id"])
            extra.append(dict(neighbor))
    return candidates + extra


# --- Détection d'intention d'énumération par type (routage par métadonnées) ---

ENUMERATION_KEYWORDS = {
    "principe": ["principe", "principes", "principle", "principles"],
    "outil": ["outil", "outils", "tool", "tools"],
    "méthode": ["méthode", "méthodes", "method", "methods"],
    "processus": ["processus", "process", "processes"],
}


def detect_content_type_intent(question: str) -> str:
    lowered = question.lower()
    for ctype, keywords in ENUMERATION_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return ctype
    return None


def is_noise_chunk(chunk: dict) -> bool:
    """Table des matières, liste des figures, index : lignes de points de
    conduite et numéros de page. Ces chunks ont été étiquetés processus/
    méthode/outil par le classifieur de la session 1 (ils contiennent les
    mots-clés), et polluent les résultats — ex. une ligne de sommaire citée
    comme si c'était la section elle-même."""
    text = chunk.get("text", "")
    return text.count("....") >= 3 or text.count(", ") > 40 and len(text) > 1500 and text.count("\n") < 3


def retrieve_by_type(content_type: str, max_results: int = 20, chunks=None,
                     question: str = None, embed_model=None, index=None,
                     include_noise: bool = False) -> list[dict]:
    """Chunks d'un content_type donné, classés par similarité à la question.

    Découvert lors de l'évaluation ("combien de méthodes d'estimation") : la
    version précédente prenait les max_results PREMIERS chunks du type dans
    l'ordre du document, sans classement. Il y a 30 chunks "méthode" ; la
    section sur l'estimation (page ~175) était au-delà des 15 premiers, et
    ce qui remontait à la place était du bruit de sommaire."""
    chunks = chunks if chunks is not None else load_chunks()
    matching = [(i, c) for i, c in enumerate(chunks)
                if c["content_type"] == content_type and (include_noise or not is_noise_chunk(c))]

    if question is None:
        return [c for _, c in matching[:max_results]]

    index = index if index is not None else load_index()
    q = embed_query(question, model=embed_model)
    scored = []
    for i, c in matching:
        v = index.reconstruct(i)          # chunk i <-> vecteur i (même ordre que chunks.json)
        c = dict(c)
        c["faiss_distance"] = float(np.sum((v - q) ** 2))
        scored.append(c)
    scored.sort(key=lambda c: c["faiss_distance"])
    return scored[:max_results]


# --- Routage spécifique aux domaines de performance ---

ALL_DOMAINS = [
    "Parties prenantes", "Équipe", "Approche de développement",
    "Planification", "Travail du projet", "Livraison", "Mesure", "Incertitude",
]

OFFICIAL_DOMAIN_NAMES = {
    "Parties prenantes": "stakeholder performance domain",
    "Équipe": "team performance domain",
    "Approche de développement": "life cycle performance domain",
    "Planification": "planning performance domain",
    "Travail du projet": "project work performance domain",
    "Livraison": "delivery performance domain",
    "Mesure": "measurement performance domain",
    "Incertitude": "uncertainty performance domain",
}

DOMAIN_ENUMERATION_MARKERS = [
    "domaines de performance", "performance domain", "domaines de la performance",
    "combien de domaines", "quels sont les domaines",
]


def is_domain_enumeration_question(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in DOMAIN_ENUMERATION_MARKERS)


def retrieve_one_per_domain(chunks=None) -> list[dict]:
    chunks = chunks if chunks is not None else load_chunks()

    found = {}
    for c in chunks:
        if c.get("content_type") != "figure":
            continue
        stripped = c["text"].strip()
        if not stripped.lower().startswith("figure"):
            continue
        lowered = stripped.lower()
        for domain_fr, official_phrase in OFFICIAL_DOMAIN_NAMES.items():
            if domain_fr in found:
                continue
            if official_phrase in lowered:
                found[domain_fr] = c

    results = []
    for d in ALL_DOMAINS:
        if d in found:
            results.append(found[d])
    return results


# --- Détection générique d'énumération (au-delà des 4 catégories fixes) ---

GENERIC_ENUMERATION_MARKERS = [
    "combien", "how many", "quels sont tous", "quels sont les",
    "liste", "list all", "énumère", "what are the", "citez",
]

BROAD_TOP_K = 12     # candidats pour les questions larges
BROAD_TOP_N = 10     # chunks gardés après reranking et envoyés au LLM
PRINCIPLES_POOL_K = 6    # candidats de la recherche hybride pour les questions sur les principes
PRINCIPLES_TYPE_K = 40   # tous les chunks "principe" (hors bruit) : le reranker choisit, pas l'embedding


def is_broad_question(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in GENERIC_ENUMERATION_MARKERS)


# --- Routage "formule" ---

import re

FORMULA_PATTERN = re.compile(
    r"\b(cpi|spi|cv|sv|ev|ac|pv|eac|bac|vac|tcpi|etc"
    r"|formules?|formulas?|calcul\w*|calculat\w*"
    r"|earned value|valeur acquise|valeur planifi\w+|co[uû]t r[ée]el)\b"
)


def is_formula_question(question: str) -> bool:
    return bool(FORMULA_PATTERN.search(question.lower()))


def retrieve_formulas(chunks=None) -> list[dict]:
    chunks = chunks if chunks is not None else load_chunks()
    return [c for c in chunks if c.get("content_type") == "formule"]


GENERIC_TERMS = set("""le la les l un une des de du d et ou en dans sur pour par au aux
ce cet cette ces que qu qui quoi quel quelle quels quelles sont est y a t il existe existent
tous toutes tout combien liste lister citez énumère mentionne mentionnés mentionnées
gestion projet projets pmbok guide édition principaux principales différents différentes
the a an of in on for to is are there what which all how many list mentioned
project management pmbok guide edition main different""".split())

CATEGORY_TERMS = {kw for kws in ENUMERATION_KEYWORDS.values() for kw in kws}


def is_pure_enumeration(question: str) -> bool:
    """Vrai si la question ne contient rien d'autre qu'un mot de catégorie
    (outils, principes, méthodes...) et des mots génériques : c'est alors une
    demande de liste complète du type, pas une question thématique."""
    tokens = _re.findall(r"[a-zà-ÿ0-9]+", question.lower())
    leftover = [t for t in tokens
                if t not in GENERIC_TERMS and t not in CATEGORY_TERMS and not t.isdigit()]
    return len(leftover) == 0


def smart_retrieve_v3(question: str, embed_model=None, index=None, chunks=None):
    """Point d'entrée principal : route la question vers l'un de 5 modes,
    du plus précis au plus général. Retourne (candidats, mode).

    Passer embed_model/index/chunks (préchargés une fois, ex: par Streamlit)
    évite de recharger le modèle d'embedding et l'index à chaque appel."""
    chunks = chunks if chunks is not None else load_chunks()

    # Vérifié EN PREMIER : "quelle méthode de calcul pour le CPI ?" contient
    # "méthode", qui déclencherait sinon le mode "type" et manquerait la formule.
    if is_formula_question(question):
        formulas = retrieve_formulas(chunks=chunks)
        if formulas:
            seen = {c["chunk_id"] for c in formulas}
            others = [c for c in retrieve(question, embed_model=embed_model, index=index, chunks=chunks)
                      if c["chunk_id"] not in seen]   # BM25 la retrouve aussi : pas de doublon
            return formulas + others, "formule"

    intent = detect_content_type_intent(question)
    if intent == "principe":
        # Le label "principe" ne sert pas ici : il a été posé par mots-clés et
        # se retrouve sur l'index, le glossaire et les annexes (4 chunks utiles
        # sur 37), alors que les principes 8 à 12 sont dans des chunks
        # étiquetés "processus". On passe donc par la recherche (hybride, sans
        # les figures qui occupaient tous les premiers rangs), on ajoute les
        # chunks qui suivent les meilleurs (la section déborde sur ses
        # voisins), et le reranker, appelé ensuite par l'application, garde les
        # BROAD_TOP_N plus pertinents pour la question.
        # Deux sources de candidats, fusionnées :
        #  - TOUS les chunks étiquetés "principe" (le label sert d'indice pour
        #    constituer un lot de candidats, pas de verdict). On ne les classe
        #    pas ici par similarité d'embedding : testé, une question française
        #    sur "12 principes" ressemble davantage à l'introduction et au
        #    glossaire qu'au chapitre 3 qui contient la liste. Le CrossEncoder,
        #    qui lit la question et le texte ensemble, est bien meilleur juge ;
        #  - quelques résultats de la recherche hybride, sans les figures.
        # include_noise=True : le SOMMAIRE (chunk 0014, "3.1 Be a Diligent...
        # 3.12 Enable Change...") est ici la meilleure source pour la liste
        # complète des 12 titres, alors que le filtre anti-bruit le rejette
        # (il compte les "...." des numéros de page). Le chapitre 3 apporte
        # ensuite les explications. Le reranker départage sommaire, chapitre,
        # index et glossaire.
        by_label = retrieve_by_type("principe", max_results=PRINCIPLES_TYPE_K, chunks=chunks,
                                    question=question, embed_model=embed_model, index=index,
                                    include_noise=True)
        hybrid = retrieve(question, k=PRINCIPLES_POOL_K, embed_model=embed_model, index=index,
                          chunks=chunks, exclude_types=("figure",))
        seen = {c["chunk_id"] for c in by_label}
        pool = by_label + [c for c in hybrid if c["chunk_id"] not in seen]
        pool = add_following_chunks(pool, chunks, top=len(by_label) + 4, span=2, exclude_types=("figure",))
        return pool, "broad"

    if intent:
        # Deux besoins opposés se cachaient derrière un même mot de catégorie.
        # "Quels sont les outils mentionnés ?" est une énumération PURE : il
        # faut TOUS les chunks du type, dans l'ordre du document (la recherche
        # sémantique n'apporte rien, elle ne fait que pousser ces chunks hors
        # du budget). "Combien de méthodes d'estimation ?" est THÉMATIQUE :
        # le mot "méthodes" est accessoire, le sujet est "estimation", et la
        # section utile (4.4.2) est étiquetée "processus" : seule la recherche
        # hybride la retrouve. Le critère : reste-t-il un terme de sujet une
        # fois retirés le mot de catégorie et les mots génériques ?
        if is_pure_enumeration(question):
            return retrieve_by_type(intent, max_results=30, chunks=chunks), "type"
        return retrieve(question, k=BROAD_TOP_K, embed_model=embed_model, index=index, chunks=chunks), "broad"

    if is_domain_enumeration_question(question):
        return retrieve_one_per_domain(chunks=chunks), "domaine"

    if is_broad_question(question):
        return retrieve(question, k=BROAD_TOP_K, embed_model=embed_model, index=index, chunks=chunks), "broad"

    return retrieve(question, embed_model=embed_model, index=index, chunks=chunks), "ciblé"


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "Comment gérer l'engagement des parties prenantes ?"
    print(f"Question : {question}\n")
    candidates, mode = smart_retrieve_v3(question)
    print(f"Mode : {mode} ({len(candidates)} candidats)\n")
    for i, r in enumerate(candidates):
        print(f"#{i + 1} — {r['chunk_id']}")