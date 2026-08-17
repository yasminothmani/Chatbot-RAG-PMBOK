"""
Structures de données partagées par tout le pipeline.

Centraliser les classes ici évite que chaque étape (extraction, chunking,
embeddings, retrieval...) redéfinisse sa propre version d'un "chunk" —
tout le monde parle le même langage de bout en bout.
"""

from dataclasses import dataclass, asdict


@dataclass
class TextBlock:
    """Une ligne de texte extraite d'un PDF, avec son contexte structurel."""
    text: str
    page: int
    source_file: str
    font_size: float
    is_heading: bool = False
    heading_level: int = 0


@dataclass
class Chunk:
    """Un chunk final, prêt à être vectorisé (session 2).
    content_type peut être 'principe' / 'processus' / 'outil' / 'méthode' (texte),
    'tableau' (issu de tables.py), ou 'figure' (issu de figures.py)."""
    text: str
    pages: list[int]
    source_file: str
    domain: str
    content_type: str
    token_count: int
    chunk_id: str = ""
    image_path: str = ""  # rempli uniquement pour content_type == "figure"

    def to_dict(self) -> dict:
        return asdict(self)
