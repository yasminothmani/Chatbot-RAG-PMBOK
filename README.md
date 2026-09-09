# RAG-PMBOK

Pipeline d'extraction, de traitement, d'indexation et de recherche pour un système RAG (Retrieval-Augmented Generation) construit sur le PMBOK 7e édition.

## Aperçu

Ce projet transforme un document PDF technique de 370 pages en une base de connaissances vectorisée, interrogeable en langage naturel. Le pipeline extrait quatre types de contenu (texte, tableaux, figures, formules), les découpe en chunks, les vectorise, les indexe, puis retrouve et classe les passages les plus pertinents pour une question posée.

## Stack technique

| Composant                    | Technologie                                             |
| ---------------------------- | ------------------------------------------------------- |
| Extraction de texte          | PyMuPDF                                                 |
| Extraction de tableaux       | Camelot                                                 |
| Rendu et traitement d'images | PyMuPDF                                                 |
| Description de figures       | OpenRouter API                                          |
| Modèle d'embedding           | paraphrase-multilingual-MiniLM-L12-v2                   |
| Base vectorielle             | FAISS (IndexFlatL2)                                     |
| Reranking                    | CrossEncoder multilingue (mmarco-mMiniLMv2-L12-H384-v1) |

## Guide de démarrage

Suis ces étapes dans l'ordre pour faire fonctionner le projet sur ta machine.

### Étape 1 : Récupérer le projet

```
git clone <url-du-repo>
cd rag-pmbok
```

### Étape 2 : Créer un environnement virtuel

Un environnement virtuel isole les dépendances de ce projet du reste de ta machine.

```
python -m venv venv
```

Active-le :

```
venv\Scripts\activate
```

Tu dois voir `(venv)` apparaître au début de ta ligne de commande.

### Étape 3 : Installer les dépendances

```
pip install -r requirements.txt
```

### Étape 4 : Configurer les clés API

Copie le fichier d'exemple :

```
copy .env.example .env
```

Ouvre `.env` et renseigne tes propres clés :

```
OPENROUTER_API_KEY=ta_cle_openrouter
GROQ_API_KEY=ta_cle_groq
```

Clé OpenRouter gratuite sur openrouter.ai. Clé Groq gratuite sur console.groq.com.

### Étape 5 : Ajouter le PDF source

Place le PDF du PMBOK 7e édition dans le dossier `data/raw/`. Le nom du fichier n'a pas d'importance, le pipeline traite tous les PDFs présents dans ce dossier.

### Étape 6 : Lancer l'extraction complète

```
cd src
python pipeline.py
```

Cette étape extrait le texte, les tableaux, les figures et les formules, puis sauvegarde le résultat dans `data/processed/chunks.json`. Compte 8 à 10 minutes, l'extraction des tableaux étant la partie la plus longue.

### Étape 7 : Générer les embeddings et construire l'index

```
python vector_store.py
```

Cette étape encode chaque chunk en vecteur, construit l'index FAISS et le sauvegarde dans `data/processed/pmbok.index`. Compte environ 30 secondes.

### Étape 8 : Tester le retrieval

```
python retrieval.py "Comment gérer l'engagement des parties prenantes ?"
```

Affiche les chunks les plus proches de la question, retrouvés par FAISS.

### Étape 9 : Tester le reranking

```
python reranker.py "Comment gérer l'engagement des parties prenantes ?"
```

Affiche l'ordre des résultats avant et après le passage par le CrossEncoder. Le premier lancement télécharge le modèle, ce qui prend un peu de temps.

## Structure du projet

```
rag-pmbok/
    requirements.txt
    .env.example
    data/
        raw/
        processed/
    src/
        config.py
        models.py
        extract.py
        chunk.py
        tables.py
        figures.py
        vision.py
        formulas.py
        multimodal.py
        pipeline.py
        embeddings.py
        vector_store.py
        retrieval.py
        reranker.py
```

## Résultats obtenus sur le corpus PMBOK 7e édition

| Type de contenu             | Nombre |
| --------------------------- | ------ |
| Chunks de texte (processus) | 258    |
| Figures                     | 61     |
| Tableaux                    | 41     |
| Chunks de texte (principe)  | 37     |
| Chunks de texte (méthode)   | 30     |
| Chunks de texte (outil)     | 13     |
| Formules                    | 1      |
| Total                       | 441    |

Chaque chunk est encodé en un vecteur de 384 dimensions. Le reranking par CrossEncoder modifie systématiquement une partie de l'ordre initial fourni par FAISS, en corrigeant les cas où la proximité vectorielle ne reflète pas exactement la pertinence réelle par rapport à la question posée.

## Choix du modèle multilingue

Le PMBOK est en anglais, mais les utilisateurs du chatbot peuvent poser leurs questions en français. Le modèle d'embedding et le modèle de reranking ont tous les deux été choisis dans leur version multilingue, plutôt que dans leur version anglaise plus répandue, pour que la recherche fonctionne correctement quelle que soit la langue de la question.

## Limitations connues

La classification par domaine repose sur une correspondance de mots-clés simple plutôt que sur un classifieur entraîné, ce qui peut produire une répartition déséquilibrée sur certains termes très fréquents dans le corpus.
