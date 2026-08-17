"""
Extraction des figures/schémas du PMBOK.

Les figures du PMBOK sont en grande majorité des graphiques VECTORIELS (formes,
flèches, diagrammes dessinés directement dans le PDF), pas des images bitmap —
impossible à extraire comme une simple image intégrée. Stratégie retenue : le
PDF contient déjà une "List of Figures and Tables" (légende + page) qu'on
parse, puis on rend la page correspondante en image PNG.

Important : l'embedding (session 2) ne "voit" jamais les pixels de l'image.
Ce qui est vectorisé, c'est le TEXTE (légende + description vision.py si
disponible) — l'image PNG sert uniquement à être affichée dans le chatbot une
fois le chunk retrouvé pertinent.
"""

import re
from dataclasses import dataclass
from pathlib import Path
import fitz

import config


@dataclass
class FigureChunk:
    caption: str
    figure_number: str
    page: int
    source_file: str
    image_path: str


FIGURE_NUM_PATTERN = re.compile(r"^Figure\s+(\d+-\d+)\.\s*$")
CAPTION_END_PATTERN = re.compile(r"^(.*?)\.{3,}\s*(\d+)\s*$")


def _find_figure_list_pages(doc: "fitz.Document", max_scan: int = 40, extra_pages: int = 6) -> list[int]:
    """Repère la 'List of Figures and Tables' et inclut les pages suivantes,
    car la liste s'étend sur plusieurs pages sans forcément répéter le titre."""
    start_pages = []
    for i in range(min(max_scan, len(doc))):
        text = doc[i].get_text("text")
        if "List of Figures and Tables" in text:
            start_pages.append(i)
    if not start_pages:
        return []
    first = start_pages[0]
    last = start_pages[-1]
    return list(range(first, min(last + extra_pages + 1, len(doc))))


def _parse_figure_entries(doc: "fitz.Document", list_pages: list[int]) -> list[tuple[str, str, int]]:
    """Extrait (numéro de figure, légende, page) depuis les pages de la liste.
    Format réel (vérifié sur le vrai PDF) réparti sur plusieurs lignes :
        'Figure 3-6.'
        'Recognize, Evaluate, and Respond '
        ''
        'to System Interactions....................37'
    On accumule jusqu'à repérer la ligne finissant par des points de suite + un nombre."""
    entries = []
    for page_idx in list_pages:
        text = doc[page_idx].get_text("text")
        lines = [l.strip() for l in text.split("\n")]

        current_fig_num = None
        caption_parts: list[str] = []

        for line in lines:
            fig_match = FIGURE_NUM_PATTERN.match(line)
            if fig_match:
                current_fig_num = f"Figure {fig_match.group(1)}"
                caption_parts = []
                continue

            if current_fig_num is None or not line:
                continue

            end_match = CAPTION_END_PATTERN.match(line)
            if end_match:
                caption_parts.append(end_match.group(1))
                page_num = int(end_match.group(2))
                full_caption = " ".join(p for p in caption_parts if p).strip()
                if full_caption:
                    entries.append((current_fig_num, full_caption, page_num))
                current_fig_num = None
                caption_parts = []
            else:
                caption_parts.append(line)

    return entries


def extract_figures(pdf_path: str, output_dir: Path, dpi: int = 150) -> list[FigureChunk]:
    """Rend chaque figure listée en image PNG et retourne un FigureChunk par figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    source_name = Path(pdf_path).name

    list_pages = _find_figure_list_pages(doc)
    if not list_pages:
        print("Aucune 'List of Figures and Tables' trouvée — extraction des figures ignorée.")
        doc.close()
        return []

    entries = _parse_figure_entries(doc, list_pages)
    print(f"{len(entries)} figures répertoriées dans la table des figures du document")

    # On ne cherche la page réelle qu'APRÈS la liste elle-même, sinon on retombe
    # systématiquement sur la page de la liste (qui mentionne aussi tous les
    # numéros de figure dans son sommaire).
    #
    # IMPORTANT : le PMBOK 7 contient deux parties distinctes ("The Standard for
    # Project Management" et "A Guide to the PMBOK") qui RECOMMENCENT chacune
    # leur propre numérotation de figures à partir de 1 — donc "Figure 2-1"
    # existe deux fois dans le document, avec un contenu différent à chaque fois.
    # Un curseur qui avance (plutôt qu'un point de départ fixe) est nécessaire :
    # comme les entrées de la liste suivent l'ordre de lecture du PDF, chercher
    # à partir de la dernière page trouvée (et non depuis le tout début à chaque
    # fois) garantit qu'on ne retombe jamais deux fois sur la même occurrence
    # d'un numéro de figure dupliqué.
    cursor = max(list_pages) + 1

    results: list[FigureChunk] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for fig_num, caption, listed_page in entries:
        actual_page = None
        for i in range(cursor, len(doc)):
            page_text = doc[i].get_text("text")
            if fig_num in page_text:
                actual_page = i
                break
        if actual_page is None:
            continue

        cursor = actual_page + 1  # la prochaine figure sera cherchée APRÈS celle-ci

        pix = doc[actual_page].get_pixmap(matrix=matrix)
        # Le numéro de figure seul ne suffit pas comme nom de fichier unique :
        # le PMBOK réutilise les mêmes numéros (Figure 2-1, 2-2...) dans ses deux
        # parties — on ajoute le numéro de page pour garantir un nom distinct et
        # éviter qu'une image n'en écrase une autre sur le disque.
        safe_name = f"{fig_num.replace(' ', '_').replace('.', '')}_p{actual_page + 1}"
        image_path = output_dir / f"{safe_name}.png"
        pix.save(str(image_path))

        results.append(FigureChunk(
            caption=caption,
            figure_number=fig_num,
            page=actual_page + 1,
            source_file=source_name,
            image_path=str(image_path),
        ))

    doc.close()
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python figures.py <chemin_vers_pdf>")
        sys.exit(1)

    figs = extract_figures(sys.argv[1], config.DATA_PROCESSED_DIR / "figures")
    print(f"\n{len(figs)} figures extraites et rendues en image\n")
    for f in figs[:10]:
        print(f"  {f.figure_number} (p.{f.page}) — {f.caption}")
        print(f"    -> {f.image_path}")