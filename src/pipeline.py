"""
Pipeline RAG-PMBOK — point d'entrée unique.

Ce fichier orchestre TOUTES les étapes du projet, dans l'ordre. C'est ici —
et nulle part ailleurs — que les sessions suivantes viennent brancher :
embeddings + indexation FAISS (session 2), retrieval + reranking (session 3),
génération avec Groq (session 4), interface Streamlit (session 5).

Usage :
    python pipeline.py
    OPENROUTER_API_KEY=ta_clé python pipeline.py   # pour activer la description
                                                     # des figures par vision LLM
"""

import json

import config
from extract import extract_all_pdfs
from chunk import build_chunks
from tables import extract_tables_by_batches
from figures import extract_figures
from multimodal import table_to_chunk, figure_to_chunk, formula_to_chunk


def run_ingestion(use_vision: bool = True) -> list:
    """Toutes les étapes d'ingestion : extraction texte, chunking, extraction
    tableaux, extraction figures (+ description vision si clé API dispo) —
    puis fusion en une seule liste de Chunks, sauvegardée pour la session 2."""

    print("=" * 60)
    print("ÉTAPE 1 — Extraction structurée des PDFs (texte)")
    print("=" * 60)
    blocks = extract_all_pdfs()
    print(f"\nTotal : {len(blocks)} blocs extraits\n")

    print("=" * 60)
    print("ÉTAPE 2 — Chunking sémantique + métadonnées (texte)")
    print("=" * 60)
    text_chunks = build_chunks(blocks)
    print(f"\nTotal : {len(text_chunks)} chunks de texte générés\n")

    print("=" * 60)
    print("ÉTAPE 3 — Extraction des tableaux")
    print("=" * 60)
    table_chunks = []
    pdf_files = sorted(config.DATA_RAW_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        tables = extract_tables_by_batches(str(pdf_path), batch_size=config.TABLE_EXTRACTION_BATCH_SIZE)
        print(f"  {pdf_path.name}: {len(tables)} tableaux détectés")
        table_chunks.extend(table_to_chunk(t, i) for i, t in enumerate(tables))
    print(f"\nTotal : {len(table_chunks)} chunks de tableaux générés\n")

    print("=" * 60)
    print("ÉTAPE 4 — Extraction des figures")
    print("=" * 60)
    figure_raw = []  # on garde les FigureChunk bruts pour l'étape vision
    figures_dir = config.DATA_PROCESSED_DIR / "figures"
    for pdf_path in pdf_files:
        figs = extract_figures(str(pdf_path), figures_dir)
        print(f"  {pdf_path.name}: {len(figs)} figures extraites")
        figure_raw.extend(figs)
    print(f"\nTotal : {len(figure_raw)} figures extraites\n")

    print("=" * 60)
    print("ÉTAPE 4bis — Description des figures par LLM à vision (OpenRouter)")
    print("=" * 60)
    descriptions: dict[str, str] = {}
    api_key = config.OPENROUTER_API_KEY
    temp_chunks = [figure_to_chunk(f, i) for i, f in enumerate(figure_raw)]
    if use_vision and api_key:
        from vision import describe_all_figures
        descriptions = describe_all_figures(temp_chunks, api_key=api_key)
        print(f"\n{len(descriptions)}/{len(figure_raw)} descriptions générées avec succès\n")
    else:
        print("Étape ignorée (pas de OPENROUTER_API_KEY dans l'environnement, ou use_vision=False).")
        print("Les figures seront indexées avec leur seule légende.\n")

    figure_chunks = [
        figure_to_chunk(f, i, description=descriptions.get(temp_chunks[i].chunk_id, ""))
        for i, f in enumerate(figure_raw)
    ]

    print("=" * 60)
    print("ÉTAPE 5 — Détection des formules mathématiques (motifs texte, sans API)")
    print("=" * 60)
    formula_chunks = []
    if config.ENABLE_FORMULAS_EXTRACTION:
        from formulas import extract_formulas
        for pdf_path in pdf_files:
            formulas = extract_formulas(str(pdf_path))
            formula_chunks.extend(formula_to_chunk(f, i) for i, f in enumerate(formulas))
        print(f"\nTotal : {len(formula_chunks)} chunks de formules générés\n")
    else:
        print("Étape désactivée (config.ENABLE_FORMULAS_EXTRACTION = False).\n")

    all_chunks = text_chunks + table_chunks + figure_chunks + formula_chunks

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in all_chunks], f, ensure_ascii=False, indent=2)
    print(f"Tous les chunks sauvegardés dans {config.CHUNKS_OUTPUT_PATH}")

    return all_chunks


if __name__ == "__main__":
    chunks = run_ingestion()

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    content_types = {}
    for c in chunks:
        content_types[c.content_type] = content_types.get(c.content_type, 0) + 1
    print("Par type de contenu :")
    for ctype, count in sorted(content_types.items(), key=lambda x: -x[1]):
        print(f"  {ctype:15s} : {count} chunks")

    domains = {}
    for c in chunks:
        domains[c.domain] = domains.get(c.domain, 0) + 1
    print("\nPar domaine :")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain:30s} : {count} chunks")