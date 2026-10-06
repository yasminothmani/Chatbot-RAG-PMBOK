# RAG-PMBOK

Un chatbot qui répond à vos questions sur le PMBOK 7e édition, en s'appuyant uniquement sur le contenu du guide, avec les sources à l'appui.

## En deux mots

On part d'un PDF de 370 pages. On en sort le texte, les tableaux, les figures et les formules, on découpe tout ça en 441 morceaux (les chunks), on les transforme en vecteurs et on les range dans un index FAISS. Quand vous posez une question, le système retrouve les morceaux les plus proches, les reclasse, les donne à un LLM avec des consignes strictes, et vous rend une réponse qui cite ses pages. Le tout dans une interface Streamlit, avec l'historique de vos conversations et le PDF consultable à côté.

## Ce qu'il y a dedans

| Étape                          | Outil                                                  |
| ------------------------------ | ------------------------------------------------------ |
| Texte du PDF                   | PyMuPDF                                                |
| Tableaux                       | Camelot                                                |
| Figures (description en texte) | modèle de vision via OpenRouter                        |
| Formules                       | expressions régulières                                 |
| Embeddings                     | paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions) |
| Index vectoriel                | FAISS, IndexFlatL2                                     |
| Recherche                      | hybride FAISS + BM25, fusion RRF                       |
| Reranking                      | CrossEncoder multilingue mmarco-mMiniLMv2-L12-H384-v1  |
| Génération                     | GPT via l'API Groq                                     |
| Interface                      | Streamlit                                              |
| Évaluation                     | RAGAS                                                  |

Tout est multilingue : le PMBOK est en anglais, les questions peuvent être en français ou en anglais, la réponse suit la langue de la question.

## Installation, pas à pas

Toutes les commandes sont pour Windows (PowerShell). Sur Mac ou Linux, remplacez `venv\Scripts\activate` par `source venv/bin/activate` et `copy` par `cp`.

### 1. Récupérer le projet

```
git clone <url-du-repo>
cd rag-pmbok
```

### 2. Créer l'environnement virtuel

Ça isole les bibliothèques du projet du reste de votre machine. À faire une seule fois.

```
python -m venv venv
venv\Scripts\activate
```

`(venv)` doit apparaître au début de la ligne. Pensez à réactiver l'environnement à chaque nouvelle fenêtre de terminal.

### 3. Installer les dépendances

```
pip install -r requirements.txt
```

Comptez quelques minutes, sentence-transformers tire PyTorch derrière lui.

### 4. Les clés API

```
copy .env.example .env
```

Ouvrez `.env` et collez vos clés :

```
OPENROUTER_API_KEY=votre_cle_openrouter
GROQ_API_KEY=votre_cle_groq
```

Les deux sont gratuites : OpenRouter sur openrouter.ai (utilisée seulement pour décrire les figures à l'extraction), Groq sur console.groq.com (utilisée pour générer les réponses et pour l'évaluation). Le fichier `.env` reste sur votre machine, il n'est jamais poussé sur Git.

### 5. Le PDF

Déposez le PMBOK 7e édition dans `data/raw/` sous le nom `pmbok7.pdf`. C'est ce nom que l'interface utilise pour afficher les pages.

### 6. Extraire le contenu

```
cd src
python pipeline.py
```

Texte, tableaux, figures et formules sont extraits et découpés en chunks dans `data/processed/chunks.json`. C'est l'étape la plus longue, 8 à 10 minutes, Camelot prend son temps sur les tableaux.

### 7. Construire l'index

```
python vector_store.py
```

Chaque chunk devient un vecteur, l'index FAISS est écrit dans `data/processed/pmbok.index`. Une trentaine de secondes.

### 8. Vérifier que ça marche en ligne de commande

Retrieval seul :

```
python retrieval.py "Comment gérer l'engagement des parties prenantes ?"
```

Retrieval + reranking :

```
python reranker.py "Comment gérer l'engagement des parties prenantes ?"
```

Réponse complète avec le LLM :

```
python generation.py "Quels sont les 12 principes du PMBOK ?"
```

Le premier lancement télécharge les modèles d'embedding et de reranking, ça prend un peu de temps, ensuite c'est instantané.

### 9. Lancer le chatbot

Toujours depuis `src/` :

```
streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`. Au premier chargement, patientez 20 à 30 secondes le temps que les modèles se chargent, puis les réponses arrivent en 2 à 4 secondes.

Ce que vous pouvez faire dans l'interface :

- poser une question, en français ou en anglais
- voir quel mode de recherche a été utilisé (badge au-dessus de la réponse)
- ouvrir les extraits cités et afficher la page du PDF correspondante
- consulter le PMBOK page par page avec le bouton en haut de la barre latérale
- retrouver vos conversations d'une fois sur l'autre, en créer de nouvelles, en supprimer

L'historique est enregistré dans `data/processed/chat_history.json`, propre à votre machine.

### 10. Évaluer le chatbot

Deux scripts, à lancer depuis `src/`.

Évaluation qualitative, 10 questions, un rapport à lire :

```
python evaluate.py
```

Résultat dans `data/processed/evaluation_results.md` : pour chaque question, le mode choisi, la langue, la durée, les sources et la réponse.

Évaluation quantitative avec RAGAS, 8 questions avec réponses de référence :

```
python evaluate_ragas.py
```

Résultat dans `data/processed/ragas_results.md`. Un LLM juge note chaque réponse sur quatre métriques entre 0 et 1 :

| Métrique          | Ce qu'elle mesure                                                         |
| ----------------- | ------------------------------------------------------------------------- |
| faithfulness      | la réponse ne dit rien qui ne soit dans les extraits (anti-hallucination) |
| answer_relevancy  | la réponse répond bien à la question posée                                |
| context_precision | les extraits utiles sont classés en tête                                  |
| context_recall    | les extraits couvrent tout ce qu'il faut pour répondre                    |

Comptez 10 à 15 minutes : le palier gratuit de Groq limite le nombre de tokens par minute, le script fait donc un appel à la fois.

## Comment le système décide quoi chercher

Toutes les questions ne se ressemblent pas, alors `retrieval.py` route chaque question vers l'un de cinq modes :

| Mode    | Déclenché par                                                          | Ce qui est fait                                              |
| ------- | ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| formule | CPI, SPI, EV, calcul...                                                | le chunk de formules est mis en tête, sans reranking         |
| type    | « quels sont les outils / principes / méthodes »                       | tous les chunks de cette catégorie, dans l'ordre du document |
| domaine | « les 8 domaines de performance »                                      | une figure par domaine, via les titres officiels             |
| broad   | « combien », « tous les », ou un mot de catégorie avec un sujet précis | recherche hybride élargie, 12 candidats, 10 gardés           |
| ciblé   | tout le reste                                                          | recherche hybride, 5 candidats, 4 gardés après reranking     |

C'est ce routage qui permet de répondre correctement à « quels sont les 12 principes ? » alors qu'une recherche vectorielle classique n'en ramènerait que quelques-uns.

## Structure du projet

```
rag-pmbok/
    requirements.txt
    .env.example
    README.md
    data/
        raw/                    pmbok7.pdf (à ajouter)
        processed/              chunks.json, pmbok.index, rapports d'évaluation
    src/
        .streamlit/config.toml  thème de l'interface
        config.py               tous les paramètres
        extract.py chunk.py tables.py figures.py vision.py formulas.py multimodal.py pipeline.py
                                extraction et découpage (session 1)
        embeddings.py vector_store.py
                                vecteurs et index FAISS (session 2)
        retrieval.py reranker.py
                                recherche hybride, routage, reranking (session 3)
        generation.py           prompt et appel au LLM (session 4)
        app.py                  interface Streamlit (session 5)
        evaluate.py evaluate_ragas.py
                                évaluation (session 6)
```

## Résultats sur le PMBOK 7

Le corpus : 441 chunks, dont 258 de texte général, 61 figures, 41 tableaux, 37 principes, 30 méthodes, 13 outils et 1 bloc de formules.

Scores RAGAS moyens sur 8 questions : faithfulness 0,95, answer_relevancy 0,76, context_precision 0,60, context_recall 0,77. Le premier est celui qui compte le plus : le chatbot n'invente pratiquement rien. Le score de précision plus bas est un choix assumé : sur les questions d'énumération, on envoie volontairement plusieurs extraits partiels pour garantir une réponse complète, et RAGAS compte ça comme du bruit.
