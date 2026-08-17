# RAG-PMBOK

Pipeline d'extraction et de traitement de documents pour un système RAG (Retrieval-Augmented Generation) construit sur le PMBOK 7e édition.

## Aperçu

Ce projet transforme un document PDF technique de 370 pages en une base de connaissances structurée, prête à être indexée pour une recherche sémantique. Le pipeline extrait et traite quatre types de contenu distincts : texte, tableaux, figures et formules mathématiques.

## Fonctionnalités

- Extraction de texte structuré avec détection automatique des titres et suppression des en-têtes/pieds de page
- Chunking sémantique basé sur la structure du document, avec chevauchement configurable
- Extraction de tableaux via reconnaissance de bordures (Camelot)
- Extraction de figures avec rendu image et description automatique par modèle de vision (OpenRouter)
- Détection de formules mathématiques par reconnaissance de motifs, sans dépendance à un service payant
- Classification automatique par domaine et type de contenu
- Sortie unifiée au format JSON, prête pour l'indexation vectorielle

## Stack technique

| Composant                             | Technologie                       |
| ------------------------------------- | --------------------------------- |
| Extraction de texte                   | PyMuPDF                           |
| Extraction de tableaux                | Camelot                           |
| Rendu et traitement d'images          | PyMuPDF                           |
| Description de figures                | OpenRouter API (modèle de vision) |
| Gestion des variables d'environnement | python-dotenv                     |

## Installation

```bash
git clone <url-du-repo>
cd rag-pmbok
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copier `.env.example` en `.env` et renseigner les clés API nécessaires :

```
OPENROUTER_API_KEY=
GROQ_API_KEY=
```

## Utilisation

Déposer le PDF source dans `data/raw/`, puis lancer le pipeline complet :

```bash
cd src
python pipeline.py
```

La sortie est générée dans `data/processed/chunks.json`, accompagnée des images de figures dans `data/processed/figures/`.

## Structure du projet

```
rag-pmbok/
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/              PDF source (non versionné)
│   └── processed/        Sorties du pipeline (non versionné)
├── src/
│   ├── config.py         Configuration centrale
│   ├── models.py         Structures de données partagées
│   ├── extract.py        Extraction de texte
│   ├── chunk.py           Chunking sémantique
│   ├── tables.py           Extraction de tableaux
│   ├── figures.py           Extraction de figures
│   ├── vision.py             Description de figures par vision
│   ├── formulas.py            Détection de formules
│   ├── multimodal.py          Unification des formats de chunk
│   └── pipeline.py            Point d'entrée principal
```

## Résultats

Sur le corpus PMBOK 7e édition (370 pages) :

| Type de contenu             | Nombre  |
| --------------------------- | ------- |
| Chunks de texte (processus) | 258     |
| Figures                     | 61      |
| Tableaux                    | 41      |
| Chunks de texte (principe)  | 37      |
| Chunks de texte (méthode)   | 30      |
| Chunks de texte (outil)     | 13      |
| Formules                    | 1       |
| **Total**                   | **441** |

Temps d'exécution estimé : 8 à 10 minutes, l'extraction de tableaux représentant la majorité du temps de traitement.

## Limitations connues

La classification par domaine repose sur une correspondance de mots-clés simple plutôt que sur un classifieur entraîné, ce qui peut produire une répartition déséquilibrée sur certains termes très fréquents dans le corpus.
