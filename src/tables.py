"""
Extraction des tableaux du PMBOK — via Camelot (outil annoncé dans le rapport).

Camelot analyse les LIGNES DE BORDURE réellement dessinées dans le PDF (mode
'lattice') pour reconstituer la structure ligne/colonne d'un tableau — plus
précis que de deviner des colonnes à partir du simple alignement du texte
(mode 'stream', testé mais écarté : beaucoup plus de faux positifs sur le
vrai PMBOK, notamment des paragraphes de texte pris pour des tableaux à
une seule colonne).
"""

from dataclasses import dataclass
from pathlib import Path
import camelot


@dataclass
class TableChunk:
    markdown: str
    page: int
    source_file: str
    n_rows: int
    n_cols: int
    accuracy: float


def _df_to_markdown(df) -> str:
    rows = df.values.tolist()
    if not rows:
        return ""
    header = [str(c).replace("\n", " ").strip() for c in rows[0]]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows[1:]:
        cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_tables(pdf_path: str, min_accuracy: float = 70.0, min_cols: int = 2, pages: str = "all") -> list[TableChunk]:
    """Parcourt le PDF avec Camelot (mode lattice) et retourne un TableChunk
    par tableau détecté avec une précision suffisante.

    pages : plage de pages Camelot (ex: 'all', '1-100', '101-370') — utile pour
    traiter un gros document par lots plutôt qu'en une seule passe très longue.

    min_accuracy filtre les détections peu fiables (score de confiance propre
    à Camelot) ; min_cols élimine les faux tableaux à une seule colonne."""
    source_name = Path(pdf_path).name
    results: list[TableChunk] = []

    tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")

    for table in tables:
        accuracy = table.parsing_report["accuracy"]
        page = table.parsing_report["page"]
        n_rows, n_cols = table.df.shape

        if accuracy < min_accuracy or n_cols < min_cols:
            continue

        md = _df_to_markdown(table.df)
        if not md.strip():
            continue

        results.append(TableChunk(
            markdown=md,
            page=page,
            source_file=source_name,
            n_rows=n_rows,
            n_cols=n_cols,
            accuracy=accuracy,
        ))

    return results


def extract_tables_by_batches(pdf_path: str, batch_size: int = 100, **kwargs) -> list[TableChunk]:
    """Découpe le document en lots de pages avant d'appeler Camelot — sur un
    PDF de 370 pages, une seule passe complète est trop longue à exécuter
    d'un bloc ; traiter par lots de ~100 pages reste rapide et donne le
    même résultat final."""
    import fitz
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    all_tables: list[TableChunk] = []
    start = 1
    while start <= n_pages:
        end = min(start + batch_size - 1, n_pages)
        page_range = f"{start}-{end}"
        print(f"    Camelot sur les pages {page_range}...")
        batch_tables = extract_tables(pdf_path, pages=page_range, **kwargs)
        all_tables.extend(batch_tables)
        start = end + 1

    return all_tables


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tables.py <chemin_vers_pdf>")
        sys.exit(1)

    tables = extract_tables(sys.argv[1])
    print(f"{len(tables)} tableaux détectés\n")
    for t in tables[:5]:
        print(f"--- Tableau p.{t.page} ({t.n_rows}x{t.n_cols}, précision {t.accuracy:.0f}%) ---")
        print(t.markdown[:400])
        print()
