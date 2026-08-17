"""
Convertit les TableChunk (tables.py) et FigureChunk (figures.py) en objets
Chunk unifiés pour qu'un tableau ou une figure soit indexé exactement comme
un chunk de texte normal (même embedding, même recherche), sans code séparé
pour chaque type de contenu en aval.
"""

from pathlib import Path

from models import Chunk
from tables import TableChunk
from figures import FigureChunk
from formulas import FormulaChunk
from chunk import _approx_tokens, _guess_domain


def table_to_chunk(t: TableChunk, index: int) -> Chunk:
    return Chunk(
        text=t.markdown,
        pages=[t.page],
        source_file=t.source_file,
        domain=_guess_domain(t.markdown),
        content_type="tableau",
        token_count=_approx_tokens(t.markdown),
        chunk_id=f"{Path(t.source_file).stem}_table_{index:04d}",
    )


def figure_to_chunk(f: FigureChunk, index: int, description: str = "") -> Chunk:
    """description : texte généré par un LLM à vision (vision.py) décrivant le
    contenu visuel de la figure. Si fourni, il est concaténé à la légende —
    c'est ce texte combiné (légende + description) qui sera vectorisé, pas les
    pixels de l'image. Sans description (vision.py non exécuté ou échoué),
    seule la légende est utilisée."""
    base_text = f"{f.figure_number}. {f.caption}"
    text = f"{base_text}\n\n{description}" if description else base_text
    return Chunk(
        text=text,
        pages=[f.page],
        source_file=f.source_file,
        domain=_guess_domain(text),
        content_type="figure",
        token_count=_approx_tokens(text),
        chunk_id=f"{Path(f.source_file).stem}_fig_{index:04d}",
        image_path=f.image_path,
    )


def formula_to_chunk(f: FormulaChunk, index: int) -> Chunk:
    return Chunk(
        text=f.text,
        pages=[f.page],
        source_file=f.source_file,
        domain=_guess_domain(f.text),
        content_type="formule",
        token_count=_approx_tokens(f.text),
        chunk_id=f"{Path(f.source_file).stem}_formula_{index:04d}",
    )
