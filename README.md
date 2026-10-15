# RAG-PMBOK

Un chatbot qui répond à vos questions sur le PMBOK 7e édition, en s'appuyant uniquement sur le contenu du guide, avec les sources à l'appui.

---

## Avant de commencer

Vous recevez le projet **avec les données déjà préparées**. L'extraction du PDF et la construction de l'index ont déjà été faites — vous n'avez pas à les relancer. Il vous suffit de déposer trois choses au bon endroit et de lancer l'application.

Il vous faut :

- Python 3.10 ou plus récent
- le fichier `pmbok7.pdf`
- les fichiers fournis : `chunks.json`, `pmbok.index`, et le dossier `figures/`
- une clé API Groq (gratuite)

Comptez 15 minutes en tout, dont 10 d'installation des bibliothèques.

Toutes les commandes sont pour Windows (PowerShell). Sur Mac ou Linux, remplacez `venv\Scripts\activate` par `source venv/bin/activate` et `copy` par `cp`.

---

## Installation, pas à pas

### Étape 1 — Récupérer le projet

```
git clone <url-du-repo>
cd rag-pmbok
```

### Étape 2 — Créer l'environnement virtuel

Ça isole les bibliothèques du projet du reste de votre machine. À faire une seule fois.

```
python -m venv venv
venv\Scripts\activate
```

`(venv)` doit apparaître au début de la ligne.

> À refaire à chaque fois que vous ouvrez une nouvelle fenêtre de terminal : `venv\Scripts\activate`

### Étape 3 — Installer les dépendances

```
pip install -r requirements.txt
```

Comptez 5 à 10 minutes, `sentence-transformers` tire PyTorch derrière lui. C'est normal que ce soit long.

### Étape 4 — La clé API

```
copy .env.example .env
```

Ouvrez le fichier `.env` et collez votre clé Groq :

```
GROQ_API_KEY=votre_cle_groq
```

La clé est gratuite, créez-la sur **console.groq.com**. Elle sert à générer les réponses.

> Il y a aussi une ligne `OPENROUTER_API_KEY` dans le fichier. **Laissez-la vide**, elle ne sert qu'à décrire les figures pendant l'extraction, et l'extraction est déjà faite.

Le fichier `.env` reste sur votre machine, il n'est jamais poussé sur Git.

### Étape 5 — Déposer le PDF

Mettez le PMBOK 7e édition ici, sous ce nom exact :

```
data/raw/pmbok7.pdf
```

Le nom compte : c'est celui que l'interface utilise pour afficher les pages citées.

### Étape 6 — Déposer les fichiers fournis

Voici les trois éléments que vous avez reçus et où ils vont :

| Ce que vous avez reçu | Où le mettre                 |
| --------------------- | ---------------------------- |
| `chunks.json`         | `data/processed/chunks.json` |
| `pmbok.index`         | `data/processed/pmbok.index` |
| le dossier `figures/` | `data/processed/figures/`    |

Si le dossier `data/processed/` n'existe pas, créez-le.

Une fois fait, vérifiez que vous avez bien ceci :

```
data/
    raw/
        pmbok7.pdf
    processed/
        chunks.json
        pmbok.index
        figures/
            ... (les images)
```

### Étape 7 — Lancer le chatbot

```
cd src
streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`.

**Au tout premier lancement**, les modèles d'embedding et de reranking se téléchargent : patientez 2 à 3 minutes. Les fois suivantes, 20 à 30 secondes de chargement, puis les réponses arrivent en 2 à 4 secondes.

C'est tout. Le chatbot fonctionne.

---

## Ce que vous pouvez faire dans l'interface

- poser une question, en français ou en anglais (la réponse suit la langue de la question)
- voir quel mode de recherche a été utilisé, via le badge au-dessus de la réponse
- ouvrir les extraits cités et afficher la page du PDF correspondante
- consulter le PMBOK page par page avec le bouton en haut de la barre latérale
- retrouver vos conversations d'une fois sur l'autre, en créer de nouvelles, en supprimer

L'historique est enregistré dans `data/processed/chat_history.json`, propre à votre machine.

---

## Si quelque chose ne marche pas

| Message d'erreur                         | Ce qu'il faut faire                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| `FileNotFoundError: chunks.json`         | le fichier n'est pas dans `data/processed/` — revoyez l'étape 6            |
| `FileNotFoundError: pmbok.index`         | pareil, l'index n'est pas au bon endroit                                   |
| le PDF ne s'affiche pas dans l'interface | le fichier doit s'appeler exactement `pmbok7.pdf` et être dans `data/raw/` |
| `GROQ_API_KEY` manquante                 | le fichier `.env` n'existe pas ou la clé n'est pas collée dedans           |
| `ModuleNotFoundError`                    | l'environnement virtuel n'est pas activé : `venv\Scripts\activate`         |
| le premier lancement semble figé         | les modèles se téléchargent, laissez tourner 2 à 3 minutes                 |

---

## Tester en ligne de commande (facultatif)

Utile pour comprendre ce que fait chaque brique. Toujours depuis `src/` :

```
python retrieval.py "Comment gérer l'engagement des parties prenantes ?"
python reranker.py "Comment gérer l'engagement des parties prenantes ?"
python generation.py "Quels sont les 12 principes du PMBOK ?"
```

La première commande montre les extraits trouvés, la deuxième les montre reclassés, la troisième donne la réponse complète du LLM.

---

## Comment ça marche

On part d'un PDF de 370 pages. On en sort le texte, les tableaux, les figures et les formules, on découpe tout ça en 441 morceaux (les chunks), on les transforme en vecteurs et on les range dans un index FAISS. Quand vous posez une question, le système retrouve les morceaux les plus proches, les reclasse, les donne à un LLM avec des consignes strictes, et vous rend une réponse qui cite ses pages.

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

### Le routage des questions

Toutes les questions ne se ressemblent pas, alors `retrieval.py` route chaque question vers l'un de cinq modes :

| Mode    | Déclenché par                                                          | Ce qui est fait                                              |
| ------- | ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| formule | CPI, SPI, EV, calcul...                                                | le chunk de formules est mis en tête, sans reranking         |
| type    | « quels sont les outils / principes / méthodes »                       | tous les chunks de cette catégorie, dans l'ordre du document |
| domaine | « les 8 domaines de performance »                                      | une figure par domaine, via les titres officiels             |
| broad   | « combien », « tous les », ou un mot de catégorie avec un sujet précis | recherche hybride élargie, 12 candidats, 10 gardés           |
| ciblé   | tout le reste                                                          | recherche hybride, 5 candidats, 4 gardés après reranking     |

C'est ce routage qui permet de répondre correctement à « quels sont les 12 principes ? » alors qu'une recherche vectorielle classique n'en ramènerait que quelques-uns.

---

## Structure du projet

```
rag-pmbok/
    requirements.txt
    .env.example
    README.md
    data/
        raw/                    pmbok7.pdf (à ajouter)
        processed/              chunks.json, pmbok.index, figures/, rapports d'évaluation
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

---

## Résultats sur le PMBOK 7

Le corpus : 441 chunks, dont 258 de texte général, 61 figures, 41 tableaux, 37 principes, 30 méthodes, 13 outils et 1 bloc de formules.

Scores RAGAS moyens sur 8 questions : faithfulness 0,95, answer_relevancy 0,76, context_precision 0,60, context_recall 0,77. Le premier est celui qui compte le plus : le chatbot n'invente pratiquement rien. Le score de précision plus bas est un choix assumé : sur les questions d'énumération, on envoie volontairement plusieurs extraits partiels pour garantir une réponse complète, et RAGAS compte ça comme du bruit.

---
