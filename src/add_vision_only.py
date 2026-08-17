"""
Script ponctuel : ajoute les descriptions vision (OpenRouter) aux figures d'un
chunks.json déjà généré, sans relancer tout le pipeline.

Usage : python add_vision_only.py
Nécessite OPENROUTER_API_KEY dans le .env.
"""

import json
import config
from vision import describe_all_figures

def main():
    with open(config.CHUNKS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    figure_chunks = [c for c in chunks if c["content_type"] == "figure"]
    print(f"{len(figure_chunks)} chunks de type 'figure' trouvés dans chunks.json")

    if not config.OPENROUTER_API_KEY:
        print("[!] OPENROUTER_API_KEY absente — vérifie ton fichier .env")
        return

    descriptions = describe_all_figures(figure_chunks, api_key=config.OPENROUTER_API_KEY)
    print(f"\n{len(descriptions)}/{len(figure_chunks)} descriptions générées")

    updated = 0
    for c in chunks:
        if c["content_type"] == "figure" and c["chunk_id"] in descriptions:
            desc = descriptions[c["chunk_id"]]
            base_text = c["text"].split("\n\n")[0]
            c["text"] = f"{base_text}\n\n{desc}"
            updated += 1

    with open(config.CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"{updated} chunks mis à jour avec leur description vision")
    print(f"chunks.json resauvegardé dans {config.CHUNKS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()