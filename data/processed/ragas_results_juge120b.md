# Évaluation RAGAS — 2026-09-29 14:57

Générateur : `openai/gpt-oss-120b` · Juge : `openai/gpt-oss-120b` · Embeddings : `paraphrase-multilingual-MiniLM-L12-v2` · 8 questions

## Scores moyens

| Métrique          | Score     | Ce que ça mesure                                                                             |
| ----------------- | --------- | -------------------------------------------------------------------------------------------- |
| faithfulness      | **0.948** | la réponse ne contient que des affirmations présentes dans les extraits (anti-hallucination) |
| answer_relevancy  | **0.762** | la réponse répond à la question posée, sans hors-sujet                                       |
| context_precision | **0.601** | les extraits pertinents sont classés en tête (retriever + reranker)                          |
| context_recall    | **0.771** | les extraits couvrent tout ce qu'il faut pour la réponse de référence (retriever)            |

## Détail par question

| #   | Question                                                                    | Mode    | Extraits | faithfulness | answer_relevancy | context_precision | context_recall |
| --- | --------------------------------------------------------------------------- | ------- | -------- | ------------ | ---------------- | ----------------- | -------------- |
| 1   | Qu'est-ce que l'approche adaptative en gestion de projet ?                  | ciblé   | 4        | 1.00         | 0.96             | 0.58              | 0.67           |
| 2   | Quels sont les outils de gestion de projet mentionnés dans le PMBOK ?       | type    | 8        | 0.91         | 0.90             | 0.14              | 1.00           |
| 3   | What are the 8 performance domains?                                         | domaine | 8        | 1.00         | 0.86             | 0.00              | 1.00           |
| 4   | Combien de méthodes d'estimation existe-t-il ?                              | broad   | 10       | 1.00         | 0.73             | 0.58              | 1.00           |
| 5   | Comment calcule-t-on le CPI et le SPI ?                                     | formule | 5        | 0.80         | 0.46             | 1.00              | 1.00           |
| 6   | Que montre la figure 2-3 sur l'engagement des parties prenantes ?           | ciblé   | 4        | 1.00         | 0.64             | 0.50              | 0.00           |
| 7   | Quelle est la différence entre un projet, un programme et un portefeuille ? | ciblé   | 4        | 1.00         | 0.76             | 1.00              | 1.00           |
| 8   | How do I calculate earned value?                                            | formule | 5        | 0.88         | 0.79             | 1.00              | 0.50           |

## Réponses générées

### 1. Qu'est-ce que l'approche adaptative en gestion de projet ?

L’approche adaptative est une méthode de gestion de projet qui repose sur des itérations successives et sur la livraison incrémentale de produits fonctionnels. Après la définition initiale du projet et de sa vision, le travail est découpé en cycles (itérations) ; chaque itération se termine par une livraison et par une priorisation du backlog à partir des retours obtenus, ce qui oriente la planification de l’itération suivante (Figure 2‑11)【pmbok7_fig_0028】.

Cette approche peut s’appuyer sur des pratiques de flux (flow‑based scheduling) – par exemple le système Kanban – qui ne suivent pas un cycle de vie en phases fixes, mais visent à optimiser le flux de livraisons, la capacité des ressources et à réduire le gaspillage de temps et de ressources (p. 0125)【pmbok7_0125】.

Lors de la sélection de l’approche de développement initiale, l’adaptative constitue l’une des trois options possibles (prédictive, hybride ou adaptative) à choisir en fonction des caractéristiques du projet (Figure 3‑2)【pmbok7_fig_0053】.

En résumé, l’approche adaptative se caractérise par :

- des itérations courtes et répétées,
- des livraisons fonctionnelles à chaque itération,
- une boucle de rétroaction continue (feedback backlog prioritization),
- une planification guidée par le flux et la capacité, souvent inspirée de Kanban.

**Sources** : pmbok7_fig_0028, pmbok7_0125, pmbok7_fig_0053.

_Référence : L'approche adaptative est une approche de développement itérative et incrémentale. Le travail est découpé en itérations qui produisent chacune un livrable fonctionnel, le backlog est re-priorisé à partir des retours obtenus, et l'approche s'appuie sur des méthodes agiles et la planification basée sur le flux. Elle convient quand les exigences sont incertaines ou évoluent souvent._

---

### 2. Quels sont les outils de gestion de projet mentionnés dans le PMBOK ?

Les extraits du PMBOK® Guide mentionnent plusieurs outils que les équipes projet peuvent utiliser :

- **Outils de communication** : mémos, courriels, rapports d’avancement, messagerie vocale (communication « push ») ; intranet, recherches Internet, dépôts en ligne (communication « pull ») ; conversations, appels téléphoniques, réunions, séances de remue‑méninges, démonstrations de produit, présentations, briefings (communication interactive) ; messagerie instantanée / textos, réseaux sociaux [Source pmbok7_0097].

- **Outils de suivi et de visualisation** : tableaux de bord (dashboards) affichant des graphiques ; graphiques « stop‑light » ou RAG (rouge‑ambre‑vert), diagrammes à barres, diagrammes circulaires (pie charts), graphiques de contrôle ; radiateurs d’information (big visible charts) [Source pmbok7_0177].

- **Outils génériques de support** : logiciels ou équipements choisis par l’équipe projet (ex. outils de planification, de suivi, etc.) [Source pmbok7_0199].

Ces éléments constituent les principaux outils de gestion de projet cités dans les passages fournis.

_Référence : Le PMBOK 7 mentionne des outils de communication (réunions, courriels, messagerie, intranet, réseaux sociaux), des tableaux de bord qui agrègent des métriques sous forme de graphiques, des radiateurs d'information (grands affichages visibles), ainsi que des modèles, méthodes et artefacts. Le choix des outils et logiciels relève de l'adaptation (tailoring) et aucun produit spécifique n'est prescrit._

---

### 3. What are the 8 performance domains?

The PMBOK 7th edition defines eight Performance Domains:

1. **Stakeholder Performance Domain** – managing stakeholder relationships and alignment. [Source: pmbok7_fig_0018]
2. **Team Performance Domain** – building a high‑performing project team and shared leadership. [Source: pmbok7_fig_0021]
3. **Development Approach and Life Cycle Performance Domain** – selecting and applying the development approach, delivery cadence, and life‑cycle phases. [Source: pmbok7_fig_0023]
4. **Planning Performance Domain** – organizing and coordinating planning activities throughout the project. [Source: pmbok7_fig_0030]
5. **Project Work Performance Domain** – establishing processes, managing physical resources, and creating a learning environment for execution. [Source: pmbok7_fig_0036]
6. **Delivery Performance Domain** – delivering the intended scope and quality to achieve benefits and stakeholder satisfaction. [Source: pmbok7_fig_0037]
7. **Measurement Performance Domain** – evaluating performance, providing actionable data, and taking corrective actions. [Source: pmbok7_fig_0040]
8. **Uncertainty Performance Domain** – recognizing, responding to, and managing project uncertainty, threats, and opportunities. [Source: pmbok7_fig_0049]

_Référence : The eight performance domains are: Stakeholder, Team, Development Approach and Life Cycle, Planning, Project Work, Delivery, Measurement, and Uncertainty._

---

### 4. Combien de méthodes d'estimation existe-t-il ?

Il y a **neuf** méthodes d’estimation répertoriées : Affinity grouping, Analogous estimating, Function points, Multipoint estimating, Parametric estimating, Relative estimating, Single‑point estimating, Story point estimation et Wideband Delphi【pmbok7_table_0035】.

_Référence : Le PMBOK 7 répertorie neuf méthodes d'estimation : le groupement par affinité, l'estimation par analogie, les points de fonction, l'estimation multipoint, l'estimation paramétrique, l'estimation relative, l'estimation à point unique, l'estimation en points d'histoire et le Wideband Delphi._

---

### 5. Comment calcule-t-on le CPI et le SPI ?

Le CPI (Cost Performance Index) se calcule en divisant la valeur acquise (EV) par le coût réel (AC) : CPI = EV / AC.  
Le SPI (Schedule Performance Index) se calcule en divisant la valeur acquise (EV) par la valeur planifiée (PV) : SPI = EV / PV.

Source : pmbok7_formula_0000.

_Référence : Le CPI (indice de performance des coûts) se calcule par CPI = EV / AC, la valeur acquise divisée par le coût réel. Le SPI (indice de performance des délais) se calcule par SPI = EV / PV, la valeur acquise divisée par la valeur planifiée._

---

### 6. Que montre la figure 2-3 sur l'engagement des parties prenantes ?

La figure 2‑3 présente un modèle cyclique d’engagement des parties prenantes : un flux continu composé de cinq étapes – **Identifier**, **Comprendre**, **Analyser**, **Prioriser** et **Engager**. Elle souligne que l’engagement est un processus itératif qui doit être surveillé et adapté tout au long du cycle de vie du projet. [Source : pmbok7_fig_0020]

_Référence : La figure 2-3 présente l'engagement efficace des parties prenantes comme un processus continu et itératif composé des étapes identifier, comprendre, analyser, prioriser, engager et suivre (monitor), répété tout au long du projet._

---

### 7. Quelle est la différence entre un projet, un programme et un portefeuille ?

**Projet**  
- Définition : « une entreprise temporaire entreprisée pour créer un produit, service ou résultat unique » ; il a un début et une fin clairement définis [ pmbok7_0028 ].  
- Caractéristiques : durée courte et temporaire, portée définie et élaborée progressivement, succès mesuré par la qualité, le respect des délais, du budget, la satisfaction du client et l’atteinte des résultats attendus [ pmbok7_table_0040 ].

**Programme**  
- Définition : « projets, sous‑programmes et activités de programme gérés de façon coordonnée pour obtenir des bénéfices qui ne seraient pas accessibles en les gérant séparément » [ pmbok7_0028 ].  
- Caractéristiques : durée plus longue, portée axée sur des bénéfices agrégés provenant de plusieurs composants, succès mesuré par la réalisation des bénéfices prévus et l’efficacité de leur livraison [ pmbok7_table_0040 ].

**Portefeuille**  
- Définition : « ensemble de projets, programmes, sous‑portefeuilles et opérations gérés comme un groupe afin d’atteindre des objectifs stratégiques » [ pmbok7_0028 ].  
- Il ne possède pas de caractéristiques détaillées dans le tableau, mais il se distingue par son rôle de regroupement de projets et programmes pour servir la stratégie globale de l’organisation.

**En résumé** : le projet est une initiative temporaire et unique, le programme regroupe plusieurs projets liés pour maximiser des bénéfices communs, et le portefeuille agrège programmes, projets et autres activités afin de réaliser les objectifs stratégiques de l’organisation.

_Sources : pmbok7_0028, pmbok7_table_0040_

_Référence : Un projet est une entreprise temporaire visant à créer un produit, service ou résultat unique. Un programme est un ensemble de projets, sous-programmes et activités liés, gérés de façon coordonnée pour obtenir des bénéfices impossibles à obtenir séparément. Un portefeuille regroupe des projets, programmes et opérations gérés collectivement pour atteindre des objectifs stratégiques._

---

### 8. How do I calculate earned value?

The excerpts you provided define earned value (EV) only indirectly—as the variable used in the cost‑variance (CV = EV – AC), schedule‑variance (SV = EV – PV), cost‑performance index (CPI = EV/AC) and schedule‑performance index (SPI = EV/PV) formulas—but they do not give a direct formula for calculating EV itself. Therefore, based on the supplied material, the method to compute earned value is not specified.

**Sources used:** pmbok7_0172 (formulas for CV, SV, CPI, SPI) and pmbok7_0172 (description of earned‑value analysis).

_Référence : Earned value (EV) is the measure of work performed expressed in terms of the budget authorized for that work. It is used in the earned value formulas: CV = EV - AC, SV = EV - PV, CPI = EV / AC and SPI = EV / PV._

---
