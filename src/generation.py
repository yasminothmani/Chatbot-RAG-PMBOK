"""
Génération de la réponse finale — dernier maillon du RAG.

Détection de langue ajoutée : le prompt système demandait au LLM de
"répondre dans la même langue que la question", mais ce type d'instruction
relative est suivi de façon peu fiable — surtout ici, où tout le reste du
prompt (labels "Extraits du PMBOK", "Question") est en français, ce qui
biaise le modèle vers le français même pour une question posée en anglais.
Solution plus robuste : détecter nous-mêmes la langue de la question, et
donner une instruction DIRECTE ("Answer in English" / "Réponds en français")
plutôt qu'une règle relative que le modèle doit déduire.
"""

import json
import re

import config

SYSTEM_PROMPT = """Tu es un expert du PMBOK 7e édition (Project Management Body of Knowledge).
Réponds à la question de l'utilisateur UNIQUEMENT à partir des extraits du PMBOK fournis ci-dessous.
Si les extraits ne permettent pas de répondre, dis-le en une phrase et ne cite AUCUNE source.
N'ajoute AUCUNE information venant de tes connaissances générales, même pour illustrer, comparer ou compléter : si ce n'est pas dans les extraits, ne l'écris pas.
Exception : le raisonnement à partir des extraits est autorisé et attendu. Si une formule est donnée, tu peux la réarranger algébriquement (si CV = EV - AC, alors EV = CV + AC) ou l'appliquer à des valeurs fournies par l'utilisateur, en montrant le calcul. Ce n'est pas une connaissance externe, c'est une déduction à partir des extraits : dis-le explicitement.
L'historique de la conversation peut t'être fourni : sers-t'en pour comprendre la question (« explique plus », « es-tu sûr ? », « et pour les programmes ? », « reformule »), mais les faits viennent toujours des extraits. Si on te demande de confirmer, relis les extraits et confirme ou corrige honnêtement.
Sois concis : va à l'essentiel, sans répéter les extraits mot pour mot.
Cite les sources utilisées (leur identifiant) à la fin de ta réponse."""

CLASSIFY_PROMPT = """Tu analyses le dernier message d'un utilisateur d'un chatbot sur le PMBOK 7, à la lumière de l'historique de la conversation.
Réponds UNIQUEMENT par un objet JSON avec ces trois champs :
- "intent" : "small_talk" si le message est une salutation, un remerciement, un au revoir, une réaction (« ok », « super ») ou une question sur le chatbot lui-même ; "follow_up" si le message demande de revenir sur la réponse précédente sans nouveau sujet (« es-tu sûr ? », « explique plus », « reformule », « un exemple », « résume », « traduis », « pourquoi ? ») ; "question" sinon.
- "language" : "fr" ou "en", la langue du message (en cas de doute, la langue de la conversation).
- "search_query" : la question reformulée en une phrase autonome, compréhensible sans l'historique, dans la langue du message, pour une recherche documentaire (« et pour les programmes ? » devient « Qu'est-ce qu'un programme ? »). Si le message est déjà autonome, le recopier tel quel. Pour small_talk et follow_up, recopier le message.
Ne réponds jamais à la question. Aucun texte en dehors du JSON."""

HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 1200

SMALL_TALK_PROMPT = (
    "Tu es l'assistant du guide PMBOK 7e édition, utilisé dans une interface de chat.\n\n"
    "Le message que tu reçois n'est pas une question sur le guide : c'est une salutation, "
    "un remerciement, un au revoir, ou une question sur toi.\n\n"
    "Réponds dans la langue du message, en une ou deux phrases, de façon naturelle et chaleureuse, "
    "sans emphase. Pas de liste, pas de markdown, pas de formule commerciale. Tu peux, si c'est "
    "pertinent, inviter la personne à poser une question sur le guide (principes, domaines de "
    "performance, méthodes, formules de la valeur acquise).\n\n"
    "INTERDIT : tu n'as aucun extrait du guide sous les yeux. Ne donne donc aucune information sur "
    "le management de projet ou le contenu du PMBOK : tout ce que tu affirmerais serait inventé. "
    "Si le message contient malgré tout une vraie question, dis simplement que tu vas chercher dans "
    "le guide et invite la personne à la poser directement."
)
# Marqueurs simples pour repérer le français — pas une vraie détection de
# langue (pas de dépendance externe ajoutée pour ça), mais suffisant pour
# distinguer une question clairement française d'une question anglaise.
FRENCH_MARKERS = [
    "é", "è", "à", "ç", "ê", "î", "ô", "û",
    "qu'est", "quels", "quelles", "combien", "comment", "pourquoi",
    "où", "qu'", " les ", " des ", " du ", " sont ", " gérer ",
    "est-ce", " le ", " la ", " un ", " une ", " et ",
]

LANGUAGE_DIRECTIVE = {
    "fr": "Réponds en français, quelle que soit la langue des extraits fournis.",
    "en": "Answer in English, regardless of the language of the provided excerpts.",
}


def detect_language(question: str) -> str:
    """Retourne 'fr' ou 'en' selon la présence de marqueurs français dans
    la question. Heuristique simple, pas un vrai détecteur de langue."""
    lowered = f" {question.lower()} "
    score = sum(1 for marker in FRENCH_MARKERS if marker in lowered)
    return "fr" if score >= 1 else "en"


# Budget de contexte en tokens. Le palier gratuit de Groq limite à 8 000
# tokens/minute (entrée + sortie) : 20 chunks de ~500 tokens dépassaient
# (erreur 413). Les chunks sont déjà classés par pertinence (similarité,
# reranking, formule épinglée) : on les garde dans cet ordre jusqu'à
# épuisement du budget. token_count est l'estimation de la session 1
# (1 token ≈ 4 caractères), sous-évaluée pour le français accentué —
# d'où une marge large. Surchargeable dans config.py (MAX_CONTEXT_TOKENS).
MAX_CONTEXT_TOKENS = getattr(config, "MAX_CONTEXT_TOKENS", 4500)


def select_chunks(chunks: list[dict], top_n: int = None) -> list[dict]:
    """Applique d'abord la limite en nombre (top_n), puis le budget en tokens.
    Le premier chunk est toujours gardé, même s'il dépasse le budget seul."""
    top_n = top_n if top_n is not None else config.TOP_N_CONTEXT
    selected, used = [], 0
    for c in chunks[:top_n]:
        n = c.get("token_count") or max(1, len(c.get("text", "")) // 4)
        if selected and used + n > MAX_CONTEXT_TOKENS:
            break
        selected.append(c)
        used += n
    return selected


def build_context(chunks: list[dict], top_n: int = None) -> str:
    selected = select_chunks(chunks, top_n)
    parts = [f"[Source: {c['chunk_id']}]\n{c['text']}" for c in selected]
    return "\n\n".join(parts)


def history_messages(history=None) -> list[dict]:
    """Les derniers échanges, au format messages, pour que le modèle comprenne
    les relances. Les réponses longues sont tronquées : l'historique sert de
    contexte, pas de source, et le budget de tokens est limité."""
    messages = []
    for turn in (history or [])[-HISTORY_TURNS:]:
        answer = turn.get("answer", "")
        if len(answer) > HISTORY_ANSWER_CHARS:
            answer = answer[:HISTORY_ANSWER_CHARS] + " […]"
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_messages(question: str, chunks: list[dict], top_n: int = None, history=None, lang: str = None) -> list[dict]:
    context = build_context(chunks, top_n)
    lang = lang or detect_language(question)
    directive = LANGUAGE_DIRECTIVE[lang]

    # La directive de langue est placee juste avant la question, pas
    # seulement dans le system prompt — les instructions proches de la fin
    # d'un prompt sont generalement mieux suivies par un LLM que des regles
    # posees plus tot et jamais rappelees.
    user_content = f"Extraits du PMBOK :\n\n{context}\n\n{directive}\n\nQuestion : {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history_messages(history),
        {"role": "user", "content": user_content},
    ]


def classify_message(message: str, history=None, client=None) -> dict:
    """Un seul appel court au LLM pour tout ce qui relève de la compréhension
    du message : salutation ou question, langue, question autonome pour la
    recherche. Remplace des listes de mots et des expressions régulières ;
    plus souple face aux fautes de frappe et aux formulations inattendues,
    au prix d'environ une demi-seconde par message. Si le JSON est illisible,
    on retombe sur un comportement sûr : question, langue par heuristique."""
    if client is None:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
    fallback = {"intent": "question", "language": detect_language(message), "search_query": message}
    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                *history_messages(history),
                {"role": "user", "content": f"Dernier message : {message}"},
            ],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            extra_body={"reasoning_effort": "low"},
        )
        raw = response.choices[0].message.content
        data = json.loads(re.search(r"\{.*\}", raw, flags=re.DOTALL).group(0))
    except Exception as e:
        print(f"[classify_message] repli sur 'question' : {e}")
        return fallback
    intent = data.get("intent") if data.get("intent") in ("small_talk", "follow_up", "question") else "question"
    language = data.get("language") if data.get("language") in ("fr", "en") else fallback["language"]
    search_query = (data.get("search_query") or "").strip() or message
    return {"intent": intent, "language": language, "search_query": search_query}


def generate_answer(question: str, chunks: list[dict], api_key: str = None, top_n: int = None, client=None,
                    history=None, lang: str = None) -> dict:
    if client is None:
        from groq import Groq
        api_key = api_key or config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante — vérifie ton fichier .env")
        client = Groq(api_key=api_key)

    messages = build_messages(question, chunks, top_n, history=history, lang=lang)

    response = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=getattr(config, "MAX_ANSWER_TOKENS", 900),  # plafonne la longueur, donc la latence
    )
    answer_text = response.choices[0].message.content

    sources = [c["chunk_id"] for c in select_chunks(chunks, top_n)]

    return {"answer": answer_text, "sources": sources}

def generate_small_talk(message: str, client=None) -> str:
    """Salutations, remerciements, au revoir : réponse courte par le LLM, avec un
    prompt qui lui interdit tout contenu sur le PMBOK puisqu'il n'a aucun extrait."""
    if client is None:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": SMALL_TALK_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0.6,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    import sys
    from retrieval import smart_retrieve_v3, BROAD_TOP_N
    from reranker import rerank

    question = sys.argv[1] if len(sys.argv) > 1 else "Comment gérer l'engagement des parties prenantes ?"
    print(f"Question : {question}\n")
    print(f"Langue detectee : {detect_language(question)}\n")

    candidates, mode = smart_retrieve_v3(question)

    if mode in ("type", "domaine"):
        result = generate_answer(question, candidates, top_n=len(candidates))
    elif mode == "formule":
        formulas = [c for c in candidates if c.get("content_type") == "formule"]
        others = rerank(question, [c for c in candidates if c.get("content_type") != "formule"])
        result = generate_answer(question, formulas + others, top_n=len(formulas) + config.TOP_N_CONTEXT)
    elif mode == "broad":
        reranked = rerank(question, candidates)
        result = generate_answer(question, reranked, top_n=BROAD_TOP_N)
    else:
        reranked = rerank(question, candidates)
        result = generate_answer(question, reranked)

    print("=" * 60)
    print(result["answer"])
    print("\nSources :", ", ".join(result["sources"]))