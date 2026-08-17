"""
Script de lecture rapide : affiche des exemples de chaque type de contenu
présent dans chunks.json (texte, tableau, figure), pour vérifier visuellement
ce qui a été extrait sans avoir à ouvrir le JSON brut.

Usage : python inspect_chunks.py [type]
  type optionnel parmi : processus, principe, methode, outil, tableau, figure, formule
  sans argument : affiche un résumé + 1 exemple de chaque type
"""

import json
import sys
import config

def main():
    with open(config.CHUNKS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    filter_type = sys.argv[1] if len(sys.argv) > 1 else None

    if filter_type:
        matching = [c for c in chunks if c["content_type"] == filter_type]
        print(f"{len(matching)} chunks de type '{filter_type}'\n")
        for c in matching:
            print(f"--- {c['chunk_id']} (pages {c['pages']}, domaine: {c['domain']}) ---")
            print(c["text"][:500])
            print()
        return

    types_seen = {}
    for c in chunks:
        types_seen.setdefault(c["content_type"], []).append(c)

    print(f"Total : {len(chunks)} chunks\n")
    for ctype, items in sorted(types_seen.items(), key=lambda x: -len(x[1])):
        print(f"=== {ctype} ({len(items)} chunks) — exemple ===")
        example = items[0]
        print(f"  {example['chunk_id']} (pages {example['pages']})")
        print(f"  {example['text'][:300]}")
        print()

if __name__ == "__main__":
    main()