# RAG-PMBOK

Pipeline d'extraction, de traitement et d'indexation de documents pour un système RAG (Retrieval-Augmented Generation) construit sur le PMBOK 7e édition.

## Aperçu

Ce projet transforme un document PDF technique de 370 pages en une base de connaissances vectorisée, prête pour la recherche sémantique. Le pipeline extrait quatre types de contenu (texte, tableaux, figures, formules), les découpe en chunks, puis les encode en vecteurs indexés pour une recherche rapide.

## Fonctionnalités

- Extraction de texte structuré avec détection automatique des titres et suppression des en-têtes/pieds de page
- Chunking sémantique basé sur la structure du document, avec chevauchement configurable
- Extraction de tableaux via reconnaissance de bordures (Camelot)
- Extraction de figures avec rendu image et description automatique par modèle de vision (OpenRouter)
- Détection de formules mathématiques par reconnaissance de motifs, sans dépendance à un service payant
- Classification automatique par domaine et type de contenu
- Génération d'embeddings multilingues et indexation vectorielle (FAISS)

## Stack technique

| Composant                             | Technologie                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| Extraction de texte                   | PyMuPDF                                                      |
| Extraction de tableaux                | Camelot                                                      |
| Rendu et traitement d'images          | PyMuPDF                                                      |
| Description de figures                | OpenRouter API (modèle de vision)                            |
| Modèle d'embedding                    | paraphrase-multilingual-MiniLM-L12-v2 (SentenceTransformers) |
| Base vectorielle                      | FAISS (IndexFlatL2)                                          |
| Gestion des variables d'environnement | python-dotenv                                                |

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

### Extraction et chunking

Déposer le PDF source dans `data/raw/`, puis lancer le pipeline d'extraction :

```bash
cd src
python pipeline.py
```

La sortie est générée dans `data/processed/chunks.json`, accompagnée des images de figures dans `data/processed/figures/`.

### Embeddings et indexation vectorielle

Une fois `chunks.json` généré, encoder les chunks et construire l'index de recherche :

```bash
python embeddings.py
python vector_store.py
```

L'index vectoriel est sauvegardé dans `data/processed/pmbok.index`.

## Structure du projet

```
rag-pmbok/
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                 PDF source (non versionné)
│   └── processed/           Sorties du pipeline (non versionné)
├── src/
│   ├── config.py            Configuration centrale
│   ├── models.py            Structures de données partagées
│   ├── extract.py           Extraction de texte
│   ├── chunk.py             Chunking sémantique
│   ├── tables.py            Extraction de tableaux
│   ├── figures.py           Extraction de figures
│   ├── vision.py            Description de figures par vision
│   ├── formulas.py          Détection de formules
│   ├── multimodal.py        Unification des formats de chunk
│   ├── pipeline.py          Point d'entrée de l'extraction
│   ├── embeddings.py        Génération des vecteurs d'embedding
│   └── vector_store.py      Construction et recherche dans l'index FAISS
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

Temps d'exécution de l'extraction : 8 à 10 minutes, l'extraction de tableaux représentant la majorité du temps de traitement. L'encodage des 441 chunks en vecteurs prend quelques secondes, sans dépendance réseau.

Chaque chunk est encodé en un vecteur de 384 dimensions, indexé dans une structure FAISS à recherche exacte (IndexFlatL2), adaptée à la taille du corpus.

## Choix du modèle d'embedding

Le modèle retenu, paraphrase-multilingual-MiniLM-L12-v2, diffère du modèle initialement envisagé (all-MiniLM-L6-v2, entraîné principalement en anglais). Ce choix répond à un besoin concret : le corpus source est en anglais, mais les descriptions de figures générées par le modèle de vision sont en français, et les utilisateurs du chatbot peuvent poser leurs questions dans l'une ou l'autre langue. Un modèle multilingue place les représentations vectorielles des deux langues dans un espace sémantique commun, permettant une recherche cohérente indépendamment de la langue de la question.

## Limitations connues

La classification par domaine repose sur une correspondance de mots-clés simple plutôt que sur un classifieur entraîné, ce qui peut produire une répartition déséquilibrée sur certains termes très fréquents dans le corpus.
