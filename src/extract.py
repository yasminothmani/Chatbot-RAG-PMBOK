"""
Session 1 — Extraction structurée des documents PDF (PMBOK, Agile Practice Guide, PMI Standards+)

Ce module lit un PDF et produit une liste de "blocs" de texte, chacun annoté
avec son niveau de titre (détecté via la taille de police), sa page d'origine,
et le nom du fichier source. C'est cette structure qui permet ensuite au
chunking (étape 2) de découper par section plutôt qu'au hasard.
"""

from pathlib import Path
from collections import Counter
import fitz  # PyMuPDF

from models import TextBlock
import config


def _most_common_font_size(doc: "fitz.Document") -> float:
    """Trouve la taille de police la plus fréquente du document = corps de texte.
    Tout ce qui est nettement plus grand est probablement un titre"""
    sizes = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span["size"])] += len(span["text"])
    if not sizes:
        return 11.0
    return sizes.most_common(1)[0][0]


def _detect_repeated_lines(doc: "fitz.Document", min_repeats: int = config.REPEATED_LINE_MIN_COUNT) -> set[str]:
    """Détecte les lignes qui reviennent sur plusieurs pages (en-têtes/pieds de page)
    pour pouvoir les filtrer elles n'apportent aucune information utile au RAG"""
    line_counts = Counter()
    for page in doc:
        text = page.get_text("text")
        for line in text.split("\n"):
            stripped = line.strip()
            if 3 < len(stripped) < 80:
                line_counts[stripped] += 1
    return {line for line, count in line_counts.items() if count >= min_repeats}

#C'est ici que tout se combine
#chaque ligne du PDF finit par être "étiquetée" page d'origine, taille de police, et statut titre/pas-titre
#C'est cette étiquette "titre" qui sera réutilisée plus tard par chunk.py pour savoir où couper
def extract_structured_blocks(pdf_path: str) -> list[TextBlock]:
    """Point d'entrée principal : ouvre un PDF et retourne une liste de TextBlock
    structurés (titres détectés, en-têtes/pieds de page supprimés)"""
    doc = fitz.open(pdf_path)
    source_name = Path(pdf_path).name
    body_size = _most_common_font_size(doc)
    repeated_lines = _detect_repeated_lines(doc)

    blocks: list[TextBlock] = []
#Parcourt chaque page puis chaque ligne de chaque page
    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                #Si la ligne est vide, ou fait partie des lignes répétées → ignorée
                #Sinon, calcule la taille de police moyenne de cette ligne
                if not line_text or line_text in repeated_lines:
                    continue  # bruit non informatif (en-tête / pied de page)

                sizes = [span["size"] for span in line.get("spans", [])]
                avg_size = sum(sizes) / len(sizes) if sizes else body_size
                #Compare à la taille de référence : si nettement plus grande (+1.5 défini dans config.py) → marquée comme titre
                is_heading = avg_size > body_size + config.HEADING_FONT_DIFF_THRESHOLD

                heading_level = 0
                if is_heading:
                    diff = avg_size - body_size
                    heading_level = 1 if diff > 6 else 2
#Crée un TextBlock avec : le texte, le numéro de page, le nom du fichier, la taille de police, et s'il s'agit d'un titre
                blocks.append(TextBlock(
                    text=line_text,
                    page=page_num,
                    source_file=source_name,
                    font_size=avg_size,
                    is_heading=is_heading,
                    heading_level=heading_level,
                ))

    doc.close()
    return blocks  #Retourne la liste complète de tous ces TextBlock


def extract_all_pdfs(raw_dir: Path = config.DATA_RAW_DIR) -> list[TextBlock]:
    """Boucle sur TOUS les PDFs présents dans data/raw/  qu'il y en ait 1, 2 ou 5
    (PMBOK seul, ou PMBOK + Agile Practice Guide, etc.), rien à changer dans le code
    quand de nouveaux documents sont ajoutés au dossier"""
    pdf_files = sorted(Path(raw_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {raw_dir} — dépose au moins un fichier avant de lancer le pipeline."
        )

    all_blocks: list[TextBlock] = []
    for pdf_path in pdf_files:
        print(f"Extraction de {pdf_path.name}...")
        blocks = extract_structured_blocks(str(pdf_path))
        print(f"  -> {len(blocks)} blocs, {sum(1 for b in blocks if b.is_heading)} titres détectés")
        all_blocks.extend(blocks)
    return all_blocks


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        # usage explicite : un seul fichier passé en argument
        result = extract_structured_blocks(sys.argv[1])
        print(f"{len(result)} blocs de texte extraits de {sys.argv[1]}")
    else:
        # usage par défaut : tous les PDFs de data/raw/
        result = extract_all_pdfs()
        print(f"\nTotal : {len(result)} blocs extraits depuis tous les PDFs de data/raw/")

    headings = [b for b in result if b.is_heading]
    print(f"{len(headings)} titres détectés au total")
    for h in headings[:10]:
        print(f"  [{h.source_file} p.{h.page}] (niveau {h.heading_level}) {h.text}")
