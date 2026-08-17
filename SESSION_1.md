# Session 1 — Extraction complète : texte + tableaux + figures + formules

## Structure du projet

```
rag-pmbok/
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── pmbok7.pdf         # le vrai PDF (370 pages)
│   └── processed/
│       ├── chunks.json       # sortie du pipeline — tout unifié
│       └── figures/          # images PNG des figures extraites
├── src/
│   ├── config.py              # tous les réglages du projet
│   ├── models.py               # structures de données partagées (TextBlock, Chunk)
│   ├── extract.py              # Étape 1 : extraction du TEXTE structuré (PyMuPDF)
│   ├── chunk.py                 # Étape 2 : chunking sémantique + métadonnées
│   ├── tables.py                 # Étape 3 : extraction des TABLEAUX (Camelot)
│   ├── figures.py                 # Étape 4 : extraction des FIGURES (rendu image + légende)
│   ├── vision.py                   # Étape 4bis : description des figures par LLM à vision (Groq)
│   ├── formulas.py                  # Étape 5 : reconnaissance de FORMULES (MathPix)
│   ├── multimodal.py               # convertit tout en Chunk unifié
│   └── pipeline.py                # POINT D'ENTRÉE UNIQUE — python3 pipeline.py
└── tests/
    └── make_sample_pdf.py
```

## Les 4 étapes d'extraction — outils et fonctionnement

### 1. Texte — `extract.py` (PyMuPDF)

- `page.get_text("dict")` retourne chaque ligne avec sa **taille de police**
- La taille la plus fréquente = corps de texte normal ; nettement plus grand = un titre
- Les lignes qui reviennent sur 3+ pages (en-têtes, pieds de page, numéros de page) sont filtrées

### 2. Tableaux — `tables.py` (**Camelot**, mode `lattice`)

- Analyse les **lignes de bordure réellement dessinées** dans le PDF pour reconstituer la grille ligne/colonne
- Testé aussi en mode `stream` (devine les colonnes par alignement de texte) mais écarté : beaucoup plus de faux positifs sur le vrai PMBOK
- Filtré par score de précision (Camelot fournit sa propre confiance par tableau, seuil ≥ 70%) et nombre de colonnes (≥ 2)
- **Camelot est lent** sur un gros PDF — le traitement se fait par lots de 100 pages (`extract_tables_by_batches`), pas en une seule passe

### 3. Figures — `figures.py` (rendu image) + `vision.py` (**Groq vision**, optionnel)

- Les schémas du PMBOK sont des graphiques **vectoriels** (formes dessinées), pas des images bitmap — pas extractibles directement
- Le PDF contient déjà une "List of Figures and Tables" (légende + page) qu'on parse, puis on **rend la page entière en image PNG**
- `vision.py` envoie ensuite chaque image à un modèle Groq capable de "voir", pour obtenir une vraie description du contenu visuel (pas juste la légende) — combinée au texte vectorisé
- **Point clé :** l'embedding ne voit jamais les pixels — seul le texte (légende + description) est vectorisé, l'image PNG sert uniquement à être affichée dans le chatbot

### 4. Formules mathématiques — `formulas.py` (**MathPix**, optionnel)

- Le PMBOK ne contient des formules (EVM, PERT) que sur une poignée de pages sur 370
- Recherche de mots-clés d'abord (gratuite, locale) pour repérer les pages candidates — **13 pages trouvées** sur le vrai PDF
- Seules ces pages sont ensuite envoyées à l'API MathPix pour la vraie reconnaissance de formule (texte + LaTeX)

## Étapes optionnelles NON TESTÉES avec un vrai appel API (important)

`vision.py` (Groq) et `formulas.py` (MathPix) nécessitent des clés API externes.
**Mon environnement de développement n'a pas d'accès réseau vers ces deux services** — le code
suit leur documentation officielle, mais je n'ai pas pu vérifier un vrai appel de bout en bout.
**C'est à toi de tester avec de vraies clés avant la session** :

```bash
export GROQ_API_KEY="ta_clé_groq"
export MATHPIX_APP_ID="ton_app_id"
export MATHPIX_APP_KEY="ta_clé_mathpix"
cd src
python3 pipeline.py
```

Sans ces clés, le pipeline tourne quand même normalement — ces deux étapes se désactivent
proprement (message clair, pas d'erreur), les figures gardent juste leur légende sans description,
et aucune formule n'est reconnue.

## Pourquoi Camelot plutôt que pdfplumber (changement important)

**Le rapport déjà déposé annonce Camelot** pour les tableaux — la première version de ce pipeline
utilisait pdfplumber par erreur de ma part. C'est corrigé : le code utilise maintenant Camelot,
pour que ce qui est montré en soutenance corresponde à ce qui est écrit dans le rapport.
Bonus : Camelot trouve plus de tableaux que pdfplumber sur le vrai PMBOK (41 contre 20).

## Résultat final sur le vrai PDF — 440 chunks au total

| Type de contenu   | Nombre                                        |
| ----------------- | --------------------------------------------- |
| processus (texte) | 258                                           |
| figure            | 61                                            |
| tableau           | 41                                            |
| principe (texte)  | 37                                            |
| méthode (texte)   | 30                                            |
| outil (texte)     | 13                                            |
| formule           | 0 (nécessite une clé MathPix pour être testé) |

## Temps d'exécution à prévoir

Camelot est l'étape la plus lente : compter **7 à 9 minutes** pour les tableaux sur les 370 pages,
plus 1-2 minutes pour le reste (texte, figures). Si tu ajoutes les clés Groq/MathPix, prévois du
temps supplémentaire pour les appels API (61 figures + 13 pages de formules, avec une pause entre
chaque appel pour ne pas dépasser les limites de débit). **C'est une étape one-time** : une fois
`chunks.json` généré, plus besoin de relancer l'extraction à chaque test des sessions suivantes.

## Limite à assumer honnêtement (classification de domaine)

`_guess_domain()` reste une heuristique par mots-clés simple — "Parties prenantes" domine
largement la classification car "stakeholder" traverse tout le référentiel. À présenter comme
limite du POC en perspectives, pas comme un résultat de classification fiable.

## Avant la session : à faire de ton côté

1. Relance `python3 pipeline.py` en entier une fois (prévoir ~10 min) pour vérifier que tout tourne chez toi
2. Teste `vision.py` et `formulas.py` avec tes vraies clés API — je n'ai pas pu le faire ici
3. Regarde `data/processed/chunks.json` et quelques tableaux/figures pour avoir des exemples concrets
