"""
Interface Streamlit du chatbot RAG-PMBOK — session 5.

Version 100 % Streamlit (conformément au rapport). Aucun CSS ne cible les
widgets Streamlit (boutons, zone de saisie) : ils gardent leur style natif,
colorés par le thème (.streamlit/config.toml). Le CSS ne s'applique qu'à
nos propres blocs HTML (bulles, avatars, bloc marque, barre du haut,
écran de chargement) — les seuls qui s'affichent de façon fiable quelle
que soit la version.
"""

import json
import re
import time
import uuid

import markdown
import streamlit as st
import streamlit.components.v1 as components

import config
from embeddings import load_chunks
from vector_store import load_index
from retrieval import smart_retrieve_v3, BROAD_TOP_N
from reranker import rerank
from generation import generate_answer, generate_small_talk, classify_message, detect_language


HISTORY_PATH = config.DATA_PROCESSED_DIR / "chat_history.json"
PDF_PATH = getattr(config, "PDF_PATH", config.DATA_RAW_DIR / "pmbok7.pdf")

SUGGESTED_QUESTIONS = [
    "Qu'est-ce que le PMBOK ?",
    "Les 12 principes ?",
    "Les 8 domaines de performance ?",
    "L'engagement des parties prenantes ?",
]


def load_history_from_disk() -> dict:
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_history_to_disk(conversations: dict):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)


@st.cache_resource(show_spinner=False)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_cross_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    from groq import Groq
    if not config.GROQ_API_KEY:
        return None
    return Groq(api_key=config.GROQ_API_KEY)


@st.cache_resource(show_spinner=False)
def get_faiss_index():
    return load_index()


@st.cache_data(show_spinner=False)
def get_chunks():
    return load_chunks()


MODE_META = {
    "conversation": {"label": "Conversation", "color": "#5B4B9E", "bg": "#EDEAF6"},
    "suivi": {"label": "Suite de la conversation", "color": "#5B4B9E", "bg": "#EDEAF6"},
    "type": {"label": "Énumération par type", "color": "#B0472F", "bg": "#FBEAE8"},
    "domaine": {"label": "Énumération de domaines", "color": "#C9862E", "bg": "#F7ECD9"},
    "broad": {"label": "Question large", "color": "#A66A1E", "bg": "#F7ECD9"},
    "ciblé": {"label": "Question ciblée", "color": "#6B6B6B", "bg": "#F0F0F0"},
    "formule": {"label": "Formules", "color": "#3B6E8F", "bg": "#E7F0F5"},
}

CONTENT_ICONS = {
    "figure": "🖼️", "tableau": "📊", "formule": "🧮",
    "processus": "📄", "principe": "📄", "méthode": "📄", "outil": "📄",
}


CITATION_PATTERN = re.compile(r"[【\[]\s*(?:Source\s*:?\s*)?((?:pmbok7_[A-Za-z0-9_]+)(?:\s*[;,]\s*pmbok7_[A-Za-z0-9_]+)*)\s*[】\]]")

REFUSAL_PATTERN = re.compile(
    r"(ne contiennent (aucune|pas d'?)|ne (permettent|fournissent) pas|aucune information|"
    r"n'est pas (présente|disponible|mentionnée)|pas d'information|impossible de répondre|"
    r"do(es)? not (contain|provide|include|mention|allow)|no information|cannot (answer|respond|be answered)|"
    r"not (available|present|specified|mentioned) in the (provided )?(excerpts|material))",
    re.IGNORECASE,
)
SOURCES_LINE_PATTERN = re.compile(r"\n*\**\s*Sources?( utilisées| used)?\s*\**\s*:.*$", re.IGNORECASE | re.DOTALL)


def is_refusal(answer: str) -> bool:
    """Le modèle dit qu'il ne peut pas répondre à partir des extraits. On exige
    une réponse courte : une réponse longue qui contient « ne contiennent pas »
    est en général une réponse partielle (« les extraits ne donnent pas X mais
    montrent Y »), avec de vraies sources à afficher."""
    return len(answer) < 320 and bool(REFUSAL_PATTERN.search(answer))

@st.cache_data(show_spinner=False)
def render_pdf_page(page_number: int, zoom: float = 1.6) -> bytes:
    """Rend une page du PDF source en image PNG, pour que l'utilisateur puisse
    vérifier la référence dans le document original. Les numéros de pages
    des chunks commencent à 1, PyMuPDF indexe à partir de 0.
    Mis en cache : une page n'est rendue qu'une fois par session."""
    import fitz
    with fitz.open(PDF_PATH) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")

def render_answer_html(text: str) -> str:
    """Le LLM répond en markdown (gras, listes, tableaux). Inséré tel quel
    dans nos bulles HTML, ce markdown n'était pas interprété : astérisques
    et barres de tableau apparaissaient bruts. On le convertit en HTML ici.
    Les références aux chunks ([pmbok7_0006], 【pmbok7_0006】) deviennent
    de petites pastilles au lieu de crochets bruts dans le texte."""
    def chip(m):
        ids = re.split(r"\s*[;,]\s*", m.group(1))
        return " ".join(f'<span class="cite">{i}</span>' for i in ids)
    if is_refusal(text):
        # Sur un refus, la ligne « Sources : … » écrite par réflexe n'a pas de sens.
        text = SOURCES_LINE_PATTERN.sub("", text).strip()
    text = CITATION_PATTERN.sub(chip, text)
    return markdown.markdown(text, extensions=["tables", "nl2br", "sane_lists"])


def answer_question(question: str, history=None) -> dict:
    """Un appel de compréhension (classify_message) décide du chemin :
    salutation, relance sur la réponse précédente, ou nouvelle question.
    Il fournit aussi la langue et une version autonome de la question pour
    la recherche. Le LLM comprend les fautes de frappe et les formulations
    inattendues ; aucune liste de mots à maintenir."""
    history = [h for h in (history or []) if h.get("mode") != "conversation"]
    groq_client = get_groq_client()
    meta = classify_message(question, history, client=groq_client)
    intent, lang, search_query = meta["intent"], meta["language"], meta["search_query"]

    # Garde-fous sur la langue. Le classifieur se trompe parfois sur un
    # message court et familier (« can u explain more »). 1) L'heuristique
    # est fiable dans un sens : un message sans aucun marqueur français est
    # anglais. 2) Une relance continue dans la langue de l'échange précédent,
    # sauf si elle demande explicitement une traduction.
    previous = history[-1] if history else None
    if detect_language(question) == "en":
        lang = "en"
    if intent == "follow_up" and previous and previous.get("language") \
            and not re.search(r"anglais|english|fran[çc]ais|french|traduis|translate", question, re.IGNORECASE):
        lang = previous["language"]

    if intent == "small_talk":
        reply = generate_small_talk(question, client=groq_client)
        return {"answer": reply, "mode": "conversation", "chunks_used": []}

    embed_model = get_embed_model()
    cross_model = get_cross_model()
    index = get_faiss_index()
    chunks = get_chunks()

    # Relance : mêmes extraits que la réponse précédente, pas de nouvelle
    # recherche. Le LLM voit l'historique et sait de quoi on parle.
    if intent == "follow_up" and previous and previous.get("chunks_used"):
        candidates = previous["chunks_used"]
        top_n = len(candidates) if previous.get("mode") in ("type", "domaine") else None
        result = generate_answer(question, candidates, top_n=top_n, client=groq_client, history=history, lang=lang)
        result["mode"] = "suivi"
        result["language"] = lang
        result["chunks_used"] = candidates
        result["search_query"] = previous.get("search_query", previous["question"])
        return result

    candidates, mode = smart_retrieve_v3(search_query, embed_model=embed_model, index=index, chunks=chunks)

    if mode in ("type", "domaine"):
        result = generate_answer(question, candidates, top_n=len(candidates), client=groq_client, history=history, lang=lang)
    elif mode == "formule":
        # Les chunks de formules restent en tete SANS reranking : le
        # CrossEncoder a le meme biais que l'embedding face a du contenu
        # symbolique et risquerait de les faire redescendre.
        formulas = [c for c in candidates if c.get("content_type") == "formule"]
        others = rerank(search_query, [c for c in candidates if c.get("content_type") != "formule"], model=cross_model)
        ordered = formulas + others
        result = generate_answer(question, ordered, top_n=len(formulas) + config.TOP_N_CONTEXT, client=groq_client, history=history, lang=lang)
        candidates = ordered
    elif mode == "broad":
        reranked = rerank(search_query, candidates, model=cross_model)
        result = generate_answer(question, reranked, top_n=BROAD_TOP_N, client=groq_client, history=history, lang=lang)
    else:
        reranked = rerank(search_query, candidates, model=cross_model)
        result = generate_answer(question, reranked, client=groq_client, history=history, lang=lang)

    result["mode"] = mode
    result["language"] = lang
    result["chunks_used"] = candidates
    result["search_query"] = search_query
    return result


def render_sources(entry: dict, key_suffix: str):
    if entry.get("mode") == "conversation" or not entry.get("chunks_used"):
        return
    # Seuls les extraits réellement cités par le modèle sont affichés. Sur un
    # refus (« les extraits ne contiennent pas… »), il n'y a rien à montrer,
    # même si le modèle a listé des identifiants par réflexe.
    if is_refusal(entry["answer"]):
        return
    cited = set(re.findall(r"pmbok7_[A-Za-z0-9_]+", entry["answer"]))
    shown = [c for c in entry["chunks_used"] if c["chunk_id"] in cited]
    if not shown:
        return

    pdf_available = PDF_PATH.exists()
    # Le même message porte la clé live_N pendant la réponse puis hist_N
    # après rechargement de la page : on ne garde que N pour que l'état
    # « page ouverte » survive au clic (qui déclenche un rechargement).
    msg_key = key_suffix.split("_")[-1]
    with st.expander(f"Voir les extraits utilisés ({len(shown)})"):
        for j, c in enumerate(shown):
            page_list = c.get("pages", [])
            pages = ", ".join(str(p) for p in page_list)
            icon = CONTENT_ICONS.get(c.get("content_type"), "📄")
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.markdown(f'<div class="source-row">{icon} <b>{c["chunk_id"]}</b> — page(s) {pages}</div>', unsafe_allow_html=True)
            with col_btn:
                toggle_key = f"pdf_{msg_key}_{c['chunk_id']}"
                if pdf_available and page_list:
                    if st.button("📖 Voir la page", key=f"btn_{toggle_key}", use_container_width=True):
                        st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
            if st.session_state.get(toggle_key, False):
                for p in page_list[:2]:
                    try:
                        label = pdf_page_label(p)
                        caption = f"PMBOK 7 — page {p} du PDF" + (f" (numérotée {label} dans le document)" if label else "")
                        st.image(render_pdf_page(p), caption=caption, use_column_width=True)
                    except Exception as e:
                        st.warning(f"Page {p} introuvable dans le PDF : {e}")
    st.download_button("⬇ Télécharger la réponse", data=entry["answer"], file_name="reponse.txt", key=f"dl_{key_suffix}")

def assistant_bubble_html(entry: dict) -> str:
    meta = MODE_META.get(entry["mode"], {"label": entry["mode"], "color": "#555", "bg": "#EEE"})
    return (
        f'<div class="msg-row assistant"><div class="avatar assistant">🤖</div>'
        f'<div class="bubble assistant">'
        f'<span class="mode-badge" style="color:{meta["color"]};background-color:{meta["bg"]}">{meta["label"]}</span>'
        f'{rewritten_hint(entry)}<br>'
        f'{render_answer_html(entry["answer"])}</div></div>'
    )


def rewritten_hint(entry: dict) -> str:
    """Quand la question a été reformulée pour la recherche, on le montre :
    l'utilisateur voit ce qui a réellement été cherché."""
    sq = entry.get("search_query")
    if sq and sq.strip().lower() != entry["question"].strip().lower() and entry.get("mode") != "suivi":
        return f'<span class="rewritten">recherche : « {sq} »</span>'
    return ""

@st.cache_data(show_spinner=False)
def pdf_page_label(page_number: int) -> str:
    """Numéro imprimé sur la page (ex. 140), tel que défini dans le PDF, s'il
    existe. Différent du numéro de page du fichier : le PMBOK 7 recommence
    sa numérotation au début du Guide. Le PDF du PMI ajoute un préfixe
    technique encodé (<FEFF...>) devant le numéro : on ne garde que le numéro."""
    import fitz
    with fitz.open(PDF_PATH) as doc:
        label = doc[page_number - 1].get_label() or ""
    label = re.sub(r"<[0-9A-Fa-f]+>", "", label).strip()
    match = re.search(r"([0-9]+|[ivxlcdm]+)$", label, flags=re.IGNORECASE)
    return match.group(1) if match else ""


@st.cache_data(show_spinner=False)
def pdf_page_count() -> int:
    import fitz
    with fitz.open(PDF_PATH) as doc:
        return doc.page_count


def render_pdf_viewer():
    """Lecteur intégré : Streamlit ne peut pas servir le PDF tel quel (il ne
    sert que des images), on affiche donc les pages rendues avec PyMuPDF,
    comme pour les sources, avec une navigation simple."""
    total = pdf_page_count()
    st.session_state.setdefault("pdf_page", 1)
    col_prev, col_num, col_next, col_close = st.columns([1, 2, 1, 2])
    with col_prev:
        if st.button("◀", help="Page précédente", use_container_width=True, disabled=st.session_state["pdf_page"] <= 1):
            st.session_state["pdf_page"] -= 1
            st.rerun()
    with col_num:
        page = st.number_input("Page du PDF", min_value=1, max_value=total,
                               value=st.session_state["pdf_page"], label_visibility="collapsed")
        if page != st.session_state["pdf_page"]:
            st.session_state["pdf_page"] = int(page)
            st.rerun()
    with col_next:
        if st.button("▶", help="Page suivante", use_container_width=True, disabled=st.session_state["pdf_page"] >= total):
            st.session_state["pdf_page"] += 1
            st.rerun()
    with col_close:
        if st.button("✕ Fermer le PDF", use_container_width=True, type="primary"):
            st.session_state["show_pdf"] = False
            st.rerun()
    p = st.session_state["pdf_page"]
    label = pdf_page_label(p)
    caption = f"PMBOK 7 — page {p} / {total} du PDF" + (f" (numérotée {label} dans le document)" if label else "")
    st.image(render_pdf_page(p), caption=caption, use_column_width=True)


def scroll_to(anchor_id: str):
    """Fait défiler la page jusqu'à l'ancre donnée. Nos bulles sont dans la
    page principale (pas dans une iframe), donc un petit script exécuté
    depuis un composant peut y accéder via window.parent. L'ancre peut ne
    pas encore être dans le DOM quand le script démarre : on réessaie
    quelques fois, et à défaut on descend en bas de la zone de chat."""
    components.html(
        f"""<script>
        (function () {{
            const doc = window.parent.document;
            let tries = 0;
            function go() {{
                const el = doc.getElementById("{anchor_id}");
                if (el) {{
                    el.scrollIntoView({{behavior: "auto", block: "start"}});
                    return;
                }}
                if (++tries < 20) {{ setTimeout(go, 100); return; }}
                const main = doc.querySelector('section.main, [data-testid="stMain"], [data-testid="stAppViewContainer"]');
                if (main) {{ main.scrollTop = main.scrollHeight; }}
            }}
            go();
        }})();
        </script>""",
        height=0,
    )


def process_question(question: str, current: dict):
    """Affiche la question, une bulle 'en train d'écrire', puis remplace cette
    bulle par la réponse SUR PLACE — sans st.rerun(), qui effaçait toute la
    page avant de la reconstruire (d'où le flash visible)."""
    anchor_id = f"msg-{uuid.uuid4().hex}"
    st.markdown(
        f'<div id="{anchor_id}" class="msg-row user"><div class="avatar user">👤</div>'
        f'<div class="bubble user">{question}</div></div>',
        unsafe_allow_html=True,
    )
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        '<div class="msg-row assistant"><div class="avatar assistant">🤖</div>'
        '<div class="bubble assistant"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>',
        unsafe_allow_html=True,
    )
    scroll_to(anchor_id)

    started = time.time()
    try:
        result = answer_question(question, history=current["messages"])
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"Une erreur est survenue : {e}")
        return
    # Une réponse très rapide ne laisse pas le temps à la bulle « … »
    # d'apparaître et semble surgir d'un coup. On garantit un minimum
    # d'affichage de l'indicateur.
    elapsed = time.time() - started
    if elapsed < 0.9:
        time.sleep(0.9 - elapsed)

    entry = {
        "question": question,
        "answer": result["answer"],
        "mode": result["mode"],
        "chunks_used": result["chunks_used"],
        "search_query": result.get("search_query", question),
        "language": result.get("language"),
    }
    typing_placeholder.markdown(assistant_bubble_html(entry), unsafe_allow_html=True)
    render_sources(entry, key_suffix=f"live_{len(current['messages'])}")

    current["messages"].append(entry)
    if current["title"] == "Nouvelle conversation":
        current["title"] = question[:40] + ("..." if len(question) > 40 else "")
    save_history_to_disk(st.session_state.conversations)
    # Pas de st.rerun() : la sidebar (titre, compteur) se mettra a jour a la
    # prochaine interaction, sans flash de page.


def delete_conversation(conv_id: str):
    st.session_state.conversations.pop(conv_id, None)
    if not st.session_state.conversations:
        new_conversation()
    elif st.session_state.current_id == conv_id:
        st.session_state.current_id = next(iter(reversed(st.session_state.conversations)))
    save_history_to_disk(st.session_state.conversations)


def new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": "Nouvelle conversation", "messages": []}
    st.session_state.current_id = conv_id
    save_history_to_disk(st.session_state.conversations)
    return conv_id


if "conversations" not in st.session_state:
    st.session_state.conversations = load_history_from_disk()
if "current_id" not in st.session_state:
    st.session_state.current_id = None
if not st.session_state.conversations:
    new_conversation()
if st.session_state.current_id not in st.session_state.conversations:
    st.session_state.current_id = next(iter(st.session_state.conversations))


st.set_page_config(page_title="Assistant PMBOK", page_icon="📘", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

.stApp { font-family: 'Inter', sans-serif; background-color: #F7F5F1; }
h1, h2, h3 { font-family: 'Poppins', sans-serif !important; }

/* Habillage Streamlit masque : bandeau colore en haut, menu Deploy */
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { background-color: transparent; }
[data-testid="stStatusWidget"] { display: none; }
footer { display: none; }

/* Pendant qu'une reponse se calcule, Streamlit estompe (opacite) tout ce
   qui etait deja affiche : effet "flou" desagreable dans un chat. On le
   neutralise, les anciens messages restent nets. */
[data-stale="true"] { opacity: 1 !important; }

.cite {
    display: inline-block; font-family: 'Inter', monospace; font-size: 0.7rem; font-weight: 600;
    color: #A66A1E; background-color: #F7ECD9; border: 1px solid #EBD9B8;
    border-radius: 6px; padding: 1px 7px; margin: 0 2px; vertical-align: middle;
}

/* ===== SIDEBAR — selecteur stable depuis des annees ===== */

.sb-brand { display: flex; align-items: center; gap: 0.75rem; padding: 0.4rem 0 1.2rem 0; }
.sb-logo {
    width: 42px; height: 42px; border-radius: 12px; background-color: #C0504D;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.sb-name { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.05rem; color: #2E2A28; }
.sb-tag { font-size: 0.75rem; color: #8A8A8A; margin-top: -2px; }
.sb-section { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #9A9A9A; margin: 1rem 0 0.5rem 0.2rem; }


/* ===== ZONE PRINCIPALE ===== */
.top-bar {
    background-color: #FFFFFF; border-radius: 30px; border: 1px solid #EDEAE2;
    box-shadow: 0 3px 12px rgba(0,0,0,0.045);
    padding: 0.8rem 1.5rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
}
.top-bar .left { display: flex; align-items: center; gap: 0.7rem; }
.top-bar .dot { width: 9px; height: 9px; border-radius: 50%; background-color: #3BAA6B; }
.top-bar .label { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 0.95rem; color: #3A2A28; }
.top-bar .pill { font-size: 0.72rem; color: #8A8A8A; background-color: #F5F3EF; border-radius: 20px; padding: 3px 10px; }

.chat-card {
    background-color: #FFFFFF; border-radius: 22px; border: 1px solid #EDEAE2;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05); padding: 1.8rem 2rem 1.2rem 2rem;
    margin-bottom: 1rem;
}

.empty-state { text-align: center; padding: 2.4rem 1rem 1.2rem 1rem; }
.empty-state .big-icon {
    width: 88px; height: 88px; border-radius: 26px;
    background-color: #FBEAE8; display: flex; align-items: center; justify-content: center;
    font-size: 42px; margin: 0 auto 1.2rem auto;
    box-shadow: 0 6px 20px rgba(192,80,77,0.18);
}
.empty-state .greeting { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.65rem; color: #2E2A28; }
.empty-state .greeting .accent { color: #C0504D; }
.empty-state .subtext { color: #8A8A8A; font-size: 0.95rem; margin-top: 0.4rem; margin-bottom: 1.6rem; }


.msg-row { display: flex; gap: 0.65rem; margin-bottom: 1rem; align-items: flex-start; }
.msg-row.user { flex-direction: row-reverse; }
.avatar {
    border-radius: 50%; width: 38px; height: 38px; min-width: 38px;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.avatar.user { background-color: #C0504D; }
.avatar.assistant { background-color: #C9862E; }
.bubble {
    border-radius: 18px; padding: 0.95rem 1.2rem; max-width: 78%;
    font-size: 0.95rem; line-height: 1.6; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.bubble.user { background-color: #FBEAE8; border: 1px solid #F0D3D0; border-top-right-radius: 5px; color: #3A2A28; }
.bubble.assistant { background-color: #FDFBF7; border: 1px solid #EDE6D8; border-top-left-radius: 5px; color: #2E2A22; }

.bubble table { border-collapse: collapse; width: 100%; margin: 0.6rem 0; font-size: 0.88rem; }
.bubble th, .bubble td { border: 1px solid #E5DFD3; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }
.bubble th { background-color: #F7ECD9; font-weight: 600; }
.bubble ul, .bubble ol { padding-left: 1.3rem; margin: 0.4rem 0; }
.bubble p { margin: 0.35rem 0; }
.rewritten { display: inline-block; margin-left: 0.5rem; font-size: 0.72rem; color: #8A8A9A; font-style: italic; }
.mode-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; border-radius: 20px; padding: 3px 11px; margin-bottom: 0.55rem; }
.source-row { background-color: #FAFAFA; border: 1px solid #F0F0F0; border-radius: 8px; padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.85rem; color: #444; }

.typing-dots span {
    display: inline-block; width: 7px; height: 7px; background-color: #C9862E; border-radius: 50%;
    animation: bounce 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce { 0%, 80%, 100% { transform: translateY(0); opacity: 0.35; } 40% { transform: translateY(-6px); opacity: 1; } }

/* ===== ECRAN DE CHARGEMENT ===== */
.loader-wrap { text-align: center; padding: 5rem 0 3rem 0; }
.loader-ring {
    width: 64px; height: 64px; margin: 0 auto 1.2rem auto; border-radius: 50%;
    border: 4px solid #F0D3D0; border-top-color: #C0504D;
    animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loader-title { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 1.1rem; color: #2E2A28; }
.loader-sub { color: #9A9A9A; font-size: 0.85rem; margin-top: 0.3rem; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

warmup_placeholder = st.empty()
if not st.session_state.get("warmed_up"):
    warmup_placeholder.markdown(
    '<div class="loader-wrap"><div class="loader-ring"></div>'
    '<div class="loader-title">Assistant PMBOK</div>'
    '<div class="loader-sub">Préparation de l\'assistant…</div></div>',
    unsafe_allow_html=True,
)
get_embed_model()
get_cross_model()
get_faiss_index()
get_chunks()
get_groq_client()
warmup_placeholder.empty()
st.session_state["warmed_up"] = True


with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-logo">📘</div>'
        '<div><div class="sb-name">Assistant PMBOK</div>'
        '<div class="sb-tag">Référentiel 7e édition</div></div></div>',
        unsafe_allow_html=True,
    )

    if PDF_PATH.exists():
        if st.button("📕 Consulter le PMBOK 7", use_container_width=True):
            st.session_state["show_pdf"] = not st.session_state.get("show_pdf", False)
            st.rerun()

    if st.button("＋  Nouvelle conversation", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    st.markdown('<div class="sb-section">Historique</div>', unsafe_allow_html=True)

    for conv_id, conv in reversed(list(st.session_state.conversations.items())):
        is_active = conv_id == st.session_state.current_id
        n_msgs = len(conv["messages"])
        label = f"{conv['title']}" + (f"  ({n_msgs})" if n_msgs else "")
        btn_type = "primary" if is_active else "secondary"
        col_open, col_del = st.columns([5, 1])
        with col_open:
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True, type=btn_type):
                st.session_state.current_id = conv_id
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{conv_id}", use_container_width=True, help="Supprimer cette conversation"):
                delete_conversation(conv_id)
                st.rerun()


current = st.session_state.conversations[st.session_state.current_id]
n_total = len(current["messages"])

st.markdown(
    '<div class="top-bar"><div class="left"><div class="dot"></div>'
    '<div class="label">Assistant PMBOK — en ligne</div></div>'
    f'<div class="pill">{n_total} échange{"s" if n_total > 1 else ""}</div></div>',
    unsafe_allow_html=True,
)
if st.session_state.get("show_pdf"):
    render_pdf_viewer()
    st.stop()

empty_state_ph = st.empty()
if not current["messages"]:
    with empty_state_ph.container():
        st.markdown(
            '<div class="empty-state"><div class="big-icon">🤖</div>'
            '<p class="greeting">Bonjour, je suis votre <span class="accent">assistant PMBOK</span></p>'
            '<p class="subtext">Posez une question sur le référentiel, ou choisissez un exemple ci-dessous</p></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, sq in enumerate(SUGGESTED_QUESTIONS):
            with cols[i]:
                if st.button(sq, key=f"suggestion_btn_{i}", use_container_width=True):
                    st.session_state["pending_question"] = sq
                    st.rerun()

for i, entry in enumerate(current["messages"]):
    st.markdown(
        f'<div class="msg-row user"><div class="avatar user">👤</div>'
        f'<div class="bubble user">{entry["question"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(assistant_bubble_html(entry), unsafe_allow_html=True)
    render_sources(entry, key_suffix=f"hist_{i}")
    if i < len(current["messages"]) - 1:
        st.divider()

question = st.chat_input("Votre question sur le PMBOK...")
pending = st.session_state.pop("pending_question", None)
question = question or pending
if question:
    empty_state_ph.empty()
    process_question(question, current)