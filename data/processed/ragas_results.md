# Évaluation RAGAS — 2026-09-29 19:31

Générateur : `openai/gpt-oss-120b` · Juge : `openai/gpt-oss-20b` · Embeddings : `paraphrase-multilingual-MiniLM-L12-v2` · 8 questions

## Scores moyens

| Métrique | Score | Ce que ça mesure |
|---|---|---|
| faithfulness | **0.919** | la réponse ne contient que des affirmations présentes dans les extraits (anti-hallucination) |
| answer_relevancy | **0.689** | la réponse répond à la question posée, sans hors-sujet |
| context_precision | **0.604** | les extraits pertinents sont classés en tête (retriever + reranker) |
| context_recall | **1.000** | les extraits couvrent tout ce qu'il faut pour la réponse de référence (retriever) |

## Détail par question

| # | Question | Mode | Extraits | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|---|
| 1 | Qu'est-ce que l'approche adaptative en gestion de projet ? | ciblé | 4 | 1.00 | 0.91 | 0.64 | 1.00 |
| 2 | Quels sont les outils de gestion de projet mentionnés dans le PMBOK ? | type | 8 | 0.57 | 0.96 | 0.20 | 1.00 |
| 3 | What are the 8 performance domains? | domaine | 8 | 0.89 | 0.85 | 0.00 | 1.00 |
| 4 | Combien de méthodes d'estimation existe-t-il ? | broad | 10 | 1.00 | 0.82 | 0.58 | 1.00 |
| 5 | Comment calcule-t-on le CPI et le SPI ? | formule | 5 | 1.00 | 0.36 | 1.00 | 1.00 |
| 6 | Que montre la figure 2-3 sur l'engagement des parties prenantes ? | ciblé | 4 | 1.00 | 0.79 | 0.50 | 1.00 |
| 7 | Quelle est la différence entre un projet, un programme et un portefeuille ? | ciblé | 4 | 0.89 | 0.82 | 0.92 | 1.00 |
| 8 | How do I calculate earned value? | formule | 5 | 1.00 | 0.00 | 1.00 | 1.00 |

## Réponses générées

### 1. Qu'est-ce que l'approche adaptative en gestion de projet ?

L’approche adaptative (ou agile) est une méthode de gestion de projet qui organise le travail en itérations successives. Après la définition initiale du projet et de sa vision, le projet est découpé en cycles courts ; chaque itération se termine par une livraison fonctionnelle et par une **priorisation du backlog** basée sur les retours obtenus, qui oriente la planification de l’itération suivante.  

Cette approche repose souvent sur un **scheduling basé sur le flux** (ex. Kanban) plutôt que sur des phases fixes, afin d’optimiser le débit des livraisons, de réduire le gaspillage de temps et de ressources, et d’ajuster continuellement le travail en fonction de la capacité, des matériaux et des retours du client.  

En résumé, l’approche adaptative se caractérise par :  
* des itérations incrémentales avec livraisons fonctionnelles,  
* une boucle de rétroaction continue (feedback backlog prioritization),  
* une planification flexible et orientée flux (ex. Kanban),  
* l’objectif d’optimiser le flux de valeur et de minimiser les gaspillages.  

Sources : [pmbok7_fig_0028] ; [pmbok7_0125] ; [pmbok7_fig_0053]

*Référence : L'approche adaptative est une approche de développement itérative et incrémentale. Le travail est découpé en itérations qui produisent chacune un livrable fonctionnel, le backlog est re-priorisé à partir des retours obtenus, et l'approche s'appuie sur des méthodes agiles et la planification basée sur le flux. Elle convient quand les exigences sont incertaines ou évoluent souvent.*

---

### 2. Quels sont les outils de gestion de projet mentionnés dans le PMBOK ?

Le PMBOK® Guide cite plusieurs **outils** (ou moyens) que les équipes de projet peuvent utiliser :

* **Outils de communication** : mémos, courriels, rapports d’avancement, messagerie vocale, messagerie instantanée / textos, réseaux sociaux, présentations, briefings, démonstrations de produit, séances de remue‑méninges, documents de projet, conversations et discussions ad‑hoc (communication « push », « pull » ou interactive)【pmbok7_0097】.  
* **Tableaux de bord (dashboards)** : affichages électroniques qui agrègent des métriques sous forme de graphiques (ex. : diagrammes RAG, histogrammes, camemberts, graphiques de contrôle) et permettent un drill‑down des données【pmbok7_0177】.  
* **Radiateurs d’information (information radiators / BVC)** : affichages physiques visibles (ex. : panneaux montrant le travail accompli, le travail restant, les risques) qui sont mis à jour fréquemment et favorisent le partage rapide d’informations【pmbok7_0177】.  
* **Logiciels ou équipements** : le texte indique que le choix d’outils logiciels ou matériels relève du « tailoring » du projet, même si aucun produit spécifique n’est nommé【pmbok7_0199】.  
* **Modèles, méthodes et artefacts** : le nouveau chapitre du guide regroupe ces éléments comme des supports de gestion de projet, sans prescrire quand ou comment les employer【pmbok7_0011】.  

Ces éléments constituent les principaux outils de gestion de projet mentionnés dans les extraits fournis.   (Sources : pmbok7_0011, pmbok7_0097, pmbok7_0177, pmbok7_0199)

*Référence : Le PMBOK 7 mentionne des outils de communication (réunions, courriels, messagerie, intranet, réseaux sociaux), des tableaux de bord qui agrègent des métriques sous forme de graphiques, des radiateurs d'information (grands affichages visibles), ainsi que des modèles, méthodes et artefacts. Le choix des outils et logiciels relève de l'adaptation (tailoring) et aucun produit spécifique n'est prescrit.*

---

### 3. What are the 8 performance domains?

The eight Performance Domains described in the PMBOK 7th edition are:

1. **Stakeholder Performance Domain** – focuses on productive stakeholder relationships, agreement on project objectives, and ensuring stakeholders are supportive or non‑disruptive. [Source: pmbok7_fig_0018]  
2. **Team Performance Domain** – addresses building a high‑performing project team, shared ownership, increased performance, and interpersonal leadership. [Source: pmbok7_fig_0021]  
3. **Development Approach and Life Cycle Performance Domain** – defines how the project will be developed, delivery cadence, and life‑cycle phases to maximize value. [Source: pmbok7_fig_0023]  
4. **Planning Performance Domain** – covers the organization and coordination activities needed for delivering project deliverables in an orderly, adaptable way. [Source: pmbok7_fig_0030]  
5. **Project Work Performance Domain** – involves establishing processes, managing physical resources, and creating a learning environment for efficient execution. [Source: pmbok7_fig_0036]  
6. **Delivery Performance Domain** – ensures the scope and quality are delivered, contributing to business objectives, benefits, and stakeholder satisfaction. [Source: pmbok7_fig_0037]  
7. **Measurement Performance Domain** – evaluates project performance, provides actionable data, and enables timely corrective actions to meet objectives. [Source: pmbok7_fig_0040]  
8. **Uncertainty Performance Domain** – promotes awareness of the project environment, proactive response to uncertainty, and management of threats and opportunities. [Source: pmbok7_fig_0049]

*Référence : The eight performance domains are: Stakeholder, Team, Development Approach and Life Cycle, Planning, Project Work, Delivery, Measurement, and Uncertainty.*

---

### 4. Combien de méthodes d'estimation existe-t-il ?

Il y a **neuf** méthodes d’estimation répertoriées : Affinity grouping, Analogous estimating, Function points, Multipoint estimating, Parametric estimating, Relative estimating, Single‑point estimating, Story point estimation et Wideband Delphi【pmbok7_table_0035】.

*Référence : Le PMBOK 7 répertorie neuf méthodes d'estimation : le groupement par affinité, l'estimation par analogie, les points de fonction, l'estimation multipoint, l'estimation paramétrique, l'estimation relative, l'estimation à point unique, l'estimation en points d'histoire et le Wideband Delphi.*

---

### 5. Comment calcule-t-on le CPI et le SPI ?

Le CPI (Cost Performance Index) se calcule en divisant la valeur acquise (EV) par le coût réel (AC) : CPI = EV / AC.  
Le SPI (Schedule Performance Index) se calcule en divisant la valeur acquise (EV) par la valeur planifiée (PV) : SPI = EV / PV.  

**Source :** [pmbok7_formula_0000]

*Référence : Le CPI (indice de performance des coûts) se calcule par CPI = EV / AC, la valeur acquise divisée par le coût réel. Le SPI (indice de performance des délais) se calcule par SPI = EV / PV, la valeur acquise divisée par la valeur planifiée.*

---

### 6. Que montre la figure 2-3 sur l'engagement des parties prenantes ?

La figure 2‑3 présente un modèle cyclique d’engagement des parties prenantes : un flux continu composé de cinq étapes – Identifier, Comprendre, Analyser, Prioriser, puis Engager. Elle souligne que l’engagement est un processus itératif, nécessitant une vigilance et une adaptation permanentes tout au long du cycle de vie du projet【pmbok7_fig_0020】.

*Référence : La figure 2-3 présente l'engagement efficace des parties prenantes comme un processus continu et itératif composé des étapes identifier, comprendre, analyser, prioriser, engager et suivre (monitor), répété tout au long du projet.*

---

### 7. Quelle est la différence entre un projet, un programme et un portefeuille ?

**Projet**  
- Définition : « une entreprise temporaire entreprendre pour créer un produit, service ou résultat unique » ; il a un début et une fin clairement définis.  
- Durée : courte, temporaire.  
- Portée : objectifs définis, élaborée progressivement tout au long du cycle de vie.  
- Gestion du changement : les équipes de projet s’attendent à des changements et mettent en place des processus pour les traiter.  
- Critère de succès : qualité du produit/projet, respect des délais et du budget, satisfaction du client et atteinte des résultats attendus.  

**Programme**  
- Définition : « projets, sous‑programmes et activités de programme gérés de façon coordonnée pour obtenir des bénéfices qui ne seraient pas disponibles en les gérant séparément ».  
- Durée : plus longue que celle d’un projet.  
- Portée : bénéfices agrégés délivrés par plusieurs composantes.  
- Gestion du changement : les équipes de programme explorent les changements et les adaptent afin d’optimiser la livraison des bénéfices.  
- Critère de succès : réalisation des bénéfices prévus et efficacité/efficience de leur livraison.  

**Portefeuille**  
- Définition : « ensemble de projets, programmes, sous‑portefeuilles et opérations gérés comme un groupe pour atteindre des objectifs stratégiques ».  
- Durée : long terme, alignée à la stratégie de l’organisation.  
- Portée : focalisée sur les bénéfices et la valeur pour les parties prenantes, au‑delà des seuls livrables.  
- Gestion du changement : les équipes de portefeuille explorent les changements pour optimiser la création de valeur globale.  
- Critère de succès : capacité à délivrer les bénéfices attendus et viabilité continue pour assurer le financement.  

Ces distinctions sont résumées dans le tableau comparatif des caractéristiques (pmbok7_table_0040) et les définitions des termes (pmbok7_0028).  

**Sources** : pmbok7_0028, pmbok7_table_0040.

*Référence : Un projet est une entreprise temporaire visant à créer un produit, service ou résultat unique. Un programme est un ensemble de projets, sous-programmes et activités liés, gérés de façon coordonnée pour obtenir des bénéfices impossibles à obtenir séparément. Un portefeuille regroupe des projets, programmes et opérations gérés collectivement pour atteindre des objectifs stratégiques.*

---

### 8. How do I calculate earned value?

The excerpts provide the formulas that use Earned Value (EV) – for example CV = EV – AC, SV = EV – PV, CPI = EV / AC, and SPI = EV / PV – but they do not give a direct definition or calculation method for EV itself. Therefore, based solely on the provided material, a specific formula for calculating Earned Value is not available.  

**Sources:** [pmbok7_formula_0000]; [pmbok7_0172]

*Référence : Earned value (EV) is the measure of work performed expressed in terms of the budget authorized for that work. It is used in the earned value formulas: CV = EV - AC, SV = EV - PV, CPI = EV / AC and SPI = EV / PV.*

---
