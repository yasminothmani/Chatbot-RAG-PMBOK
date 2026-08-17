"""
Détection des formules mathématiques du PMBOK (EVM, PERT...).

Vérification faite sur le vrai PDF avant de construire ce module : les
formules du PMBOK 7 (CV = EV - AC, CPI = EV / AC...) sont écrites en TEXTE
NORMAL dans le PDF, pas comme des équations typographiques à part — elles
sont donc déjà capturées par extract.py, au même titre que n'importe quel
autre texte de la page.

Deux outils avaient été envisagés puis écartés pour cette raison :
- MathPix : nécessite un compte payant (carte bancaire dès l'inscription),
  et aurait été inutile ici puisque le texte est déjà exploitable tel quel
- pix2tex (LaTeX-OCR, alternative locale/gratuite) : conçu pour reconnaître
  une équation image isolée, pas adapté à une page entière de texte courant

Ce module se contente donc de repérer, dans le texte déjà extrait, les
passages qui RESSEMBLENT à une formule (motif "XXX = ...") et de les
regrouper en chunks dédiés avec le type 'formule' — pour qu'ils soient
facilement identifiables et priorisables au retrieval, plutôt que noyés
dans un chunk de texte général.
"""

import re
from dataclasses import dataclass
from pathlib import Path
import fitz

# Motif : 2 à 5 lettres majuscules, suivi de '=', ex: "CPI = EV / AC"
FORMULA_LINE_PATTERN = re.compile(r'^[A-Z]{2,5}\s*=\s*.+$')


@dataclass
class FormulaChunk:
    text: str
    page: int
    source_file: str
    formulas_found: list[str]


def _extract_formula_lines(page_text: str) -> list[str]:
    """Repère les lignes qui correspondent au motif d'une formule PMBOK."""
    lines = [l.strip() for l in page_text.split("\n")]
    return [l for l in lines if FORMULA_LINE_PATTERN.match(l)]


def extract_formulas(pdf_path: str) -> list[FormulaChunk]:
    """Parcourt tout le document et retourne un FormulaChunk par page
    contenant au moins une ligne ressemblant à une formule."""
    source_name = Path(pdf_path).name
    doc = fitz.open(pdf_path)
    results: list[FormulaChunk] = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        formulas = _extract_formula_lines(text)
        if not formulas:
            continue

        combined = "\n".join(formulas)

        results.append(FormulaChunk(
            text=combined,
            page=i + 1,
            source_file=source_name,
            formulas_found=formulas,
        ))

    doc.close()
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python formulas.py <chemin_vers_pdf>")
        sys.exit(1)

    formulas = extract_formulas(sys.argv[1])
    print(f"{len(formulas)} pages avec formules détectées\n")
    for f in formulas:
        print(f"--- Page {f.page} ---")
        for line in f.formulas_found:
            print(f"  {line}")
        print()