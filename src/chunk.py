"""
Chunking sémantique + métadonnées.

Prend la liste de TextBlock produite par extract.py et regroupe les lignes
en chunks d'environ 512 tokens (avec chevauchement de 50 tokens), en coupant
aux frontières de section MAJEURES (titres de niveau 1) — pas à chaque
sous-titre, sinon les chunks deviennent beaucoup trop petits (testé sur le
vrai PMBOK : couper à chaque sous-titre donnait 1019 chunks de 75 tokens en
moyenne au lieu de 338 chunks de ~519 tokens, largement trop fragmenté).

Chaque chunk final porte des métadonnées : domaine PMBOK, page(s) d'origine,
type de contenu, fichier source.
"""

from pathlib import Path

from models import TextBlock, Chunk
import config

TARGET_TOKENS = config.CHUNK_TARGET_TOKENS
OVERLAP_TOKENS = config.CHUNK_OVERLAP_TOKENS

# Mots-clés pour rattacher un chunk à un domaine PMBOK. Le PMBOK 7 est en anglais
# (édition originale PMI) — les mots-clés couvrent donc l'anglais en priorité.
# Note : "stakeholder" domine largement la classification sur le vrai PMBOK
# (ce terme traverse presque tout le référentiel) — c'est une limite connue de
# cette heuristique simple, assumée comme telle dans le rapport.
DOMAIN_KEYWORDS = {
    "Parties prenantes": ["stakeholder", "partie prenante"],
    "Équipe": ["team performance", "leadership", "équipe projet"],
    "Approche de développement": ["development approach", "agile", "scrum", "kanban", "predictive", "adaptive life cycle"],
    "Planification": ["planning", "planification", "schedule", "échéancier", "wbs", "estimate"],
    "Travail du projet": ["project work", "travail du projet", "resource"],
    "Livraison": ["delivery", "livraison", "deliverable", "livrable", "quality"],
    "Mesure": ["measurement", "mesure", "performance domain", "metric", "kpi"],
    "Incertitude": ["risk", "risque", "uncertainty", "incertitude", "threat", "ambiguity"],
}

CONTENT_TYPE_KEYWORDS = {
    "principe": ["principle", "principe"],
    "processus": ["process", "processus"],
    "outil": ["tool", "outil", "template", "modèle"],
    "méthode": ["method", "méthode", "technique"],
}


def _approx_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (1 token ≈ 4 caractères).
    Suffisant pour du dimensionnement de chunk, pas besoin d'un vrai tokenizer."""
    return max(1, len(text) // 4)


def _guess_domain(text: str) -> str:
    lowered = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return domain
    return "Non classé"


def _guess_content_type(text: str) -> str:
    lowered = text.lower()
    for ctype, keywords in CONTENT_TYPE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return ctype
    return "processus"


def build_chunks(blocks: list[TextBlock]) -> list[Chunk]:
    """Découpe une liste de TextBlock (potentiellement issue de PLUSIEURS PDFs)
    en chunks. Un changement de fichier source force toujours la fermeture du
    chunk en cours — on ne mélange jamais deux documents différents."""
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_pages: set[int] = set()
    current_source: str = blocks[0].source_file if blocks else "unknown"

    def flush(source_file: str):
        if not current_lines:
            return
        text = " ".join(current_lines).strip()
        if not text:
            return
        chunk_id = f"{Path(source_file).stem}_{len(chunks):04d}"
        chunks.append(Chunk(
            text=text,
            pages=sorted(current_pages),
            source_file=source_file,
            domain=_guess_domain(text),
            content_type=_guess_content_type(text),
            token_count=_approx_tokens(text),
            chunk_id=chunk_id,
        ))

    for block in blocks:
        new_document = block.source_file != current_source
        current_size = _approx_tokens(" ".join(current_lines)) if current_lines else 0
        new_section = block.is_heading and block.heading_level == 1 and current_lines and current_size >= 100

        if (new_document or new_section) and current_lines:
            flush(current_source)
            if new_document:
                current_lines = []
            else:
                overlap_text = " ".join(current_lines)[-OVERLAP_TOKENS * 4:]
                current_lines = [overlap_text] if overlap_text else []
            current_pages = {block.page}
            current_source = block.source_file

        current_lines.append(block.text)
        current_pages.add(block.page)
        current_source = block.source_file

        if _approx_tokens(" ".join(current_lines)) >= TARGET_TOKENS:
            flush(current_source)
            overlap_text = " ".join(current_lines)[-OVERLAP_TOKENS * 4:]
            current_lines = [overlap_text] if overlap_text else []
            current_pages = {block.page}

    flush(current_source)
    return chunks


if __name__ == "__main__":
    import sys
    from extract import extract_structured_blocks, extract_all_pdfs

    if len(sys.argv) >= 2:
        blocks = extract_structured_blocks(sys.argv[1])
    else:
        blocks = extract_all_pdfs()

    chunks = build_chunks(blocks)

    print(f"\n{len(chunks)} chunks générés (cible ~{TARGET_TOKENS} tokens, chevauchement {OVERLAP_TOKENS})\n")
    for i, c in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ({c.chunk_id}) ---")
        print(f"  Source      : {c.source_file}")
        print(f"  Pages       : {c.pages}")
        print(f"  Domaine     : {c.domain}")
        print(f"  Type        : {c.content_type}")
        print(f"  ~Tokens     : {c.token_count}")
        print(f"  Extrait     : {c.text[:150]}...")
        print()
