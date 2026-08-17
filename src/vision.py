"""
Génère une description textuelle de chaque figure via un LLM à vision, en
passant par OpenRouter — un agrégateur qui donne accès à plusieurs modèles
(dont certains gratuits) derrière une seule API au format OpenAI standard.

Historique : ce module a d'abord utilisé Groq (modèle vision retiré du
catalogue en cours de route), puis Google Gemini (quota gratuit à 0 sur le
compte testé). Bascule sur OpenRouter pour plus de flexibilité — si un modèle
gratuit particulier n'est plus disponible, il suffit de changer le nom du
modèle dans config.py sans retoucher au reste du code.

IMPORTANT : le nom du modèle (OPENROUTER_VISION_MODEL_NAME dans config.py)
doit être vérifié sur https://openrouter.ai/models (filtrer par "vision" et
trier par prix pour voir les modèles gratuits actuels) — ces catalogues
changent vite, comme on vient de le voir deux fois de suite.

Cette étape ne tourne QU'UNE FOIS, pendant l'ingestion (pas à chaque question
posée par l'utilisateur) — le coût et la latence de l'appel vision n'affectent
donc jamais la vitesse du chatbot final.
"""

import base64
import time

import requests

import config

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

VISION_PROMPT = """Tu es un expert en gestion de projet, spécialiste du PMBOK 7e édition
(le référentiel de référence du PMI — 12 principes, 8 domaines de performance).

Tu regardes une figure extraite du PMBOK, intitulée : "{caption}"

Cette description sera indexée dans un système de recherche (RAG) pour un chatbot
qui répond à des questions sur le PMBOK. Décris en 2 à 4 phrases ce que ce schéma
montre concrètement : ses éléments principaux, leur organisation (liste, cycle,
flux, comparaison...), et l'idée clé qu'il illustre.

Utilise le vocabulaire PMBOK quand il est pertinent (ex : nommer un principe ou
un domaine de performance si la figure s'y rapporte clairement), pour que la
description soit facilement retrouvée par quelqu'un qui poserait une question sur
ce concept avec ses propres mots. Sois factuel et précis, en français. Ne décris
pas la mise en page générale de la page (titres de section, numéro de page) —
uniquement le contenu du schéma lui-même."""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_figure(image_path: str, caption: str, api_key: str) -> str:
    """Envoie une figure à un modèle de vision via OpenRouter et retourne sa
    description textuelle. En cas d'erreur, affiche le détail renvoyé par
    l'API pour diagnostiquer, et retourne une chaîne vide."""
    try:
        b64_image = _encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.OPENROUTER_VISION_MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT.format(caption=caption)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }],
            "max_tokens": 300,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        if not response.ok:
            print(f"    [debug] Statut {response.status_code} — réponse brute : {response.text[:500]}")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    [!] Échec description vision pour {image_path}: {e}")
        return ""


def describe_all_figures(figure_chunks: list, api_key: str = None, delay_seconds: float = 1.0) -> dict:
    """Prend la liste des chunks de type 'figure' et retourne un dict
    {chunk_id: description}. Nécessite une clé API OpenRouter valide."""
    api_key = api_key or config.OPENROUTER_API_KEY
    if not api_key:
        print("[!] Aucune clé OPENROUTER_API_KEY trouvée dans l'environnement — étape ignorée.")
        print("    Pour l'activer : ajoute OPENROUTER_API_KEY=ta_clé dans le fichier .env")
        return {}

    descriptions = {}

    for i, chunk in enumerate(figure_chunks):
        image_path = chunk.get("image_path") if isinstance(chunk, dict) else chunk.image_path
        caption = chunk.get("text") if isinstance(chunk, dict) else chunk.text
        chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else chunk.chunk_id

        if not image_path:
            continue

        print(f"  [{i+1}/{len(figure_chunks)}] Description de {chunk_id}...")
        desc = describe_figure(image_path, caption, api_key)
        if desc:
            descriptions[chunk_id] = desc
        time.sleep(delay_seconds)

    return descriptions


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vision.py <chemin_vers_une_image.png> [légende optionnelle]")
        sys.exit(1)

    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        print("Définis OPENROUTER_API_KEY dans ton .env avant de tester ce script.")
        sys.exit(1)

    caption = sys.argv[2] if len(sys.argv) > 2 else "Figure sans légende connue"
    description = describe_figure(sys.argv[1], caption, api_key)
    print(f"\nDescription générée :\n{description}")