"""
Génère un petit PDF de test avec titres + paragraphes, pour valider le pipeline
d'extraction/chunking AVANT d'avoir le vrai PDF du PMBOK entre les mains.
"""

import fitz

def make_sample_pdf(output_path: str):
    doc = fitz.open()

    content = [
        ("Domaine de performance : Incertitude", 16, [
            "Ce domaine de performance traite des activités et fonctions associées "
            "à la gestion du risque et de l'incertitude au sein d'un projet.",
            "Un risque est un événement ou une condition incertaine qui, s'il se produit, "
            "a un effet positif ou négatif sur un ou plusieurs objectifs du projet.",
        ]),
        ("Principe : Adaptabilité et résilience", 16, [
            "Ce principe encourage les équipes projet à intégrer l'adaptabilité et la "
            "résilience dans leurs approches, afin d'aider l'organisation à s'adapter "
            "à un environnement changeant, à se remettre des revers et à progresser.",
            "La résilience permet à l'équipe projet d'absorber les chocs et de continuer "
            "à fonctionner malgré des perturbations.",
        ]),
        ("Processus : Planification de la gestion des risques", 16, [
            "Ce processus définit comment mener les activités de gestion des risques "
            "d'un projet. Il est réalisé une fois ou à des moments prédéfinis.",
            "L'outil principal de ce processus est le registre des risques, qui "
            "consigne les résultats de l'identification, de l'analyse qualitative "
            "et de la planification des réponses.",
        ]),
    ]

    for i, (heading, size, paragraphs) in enumerate(content):
        page = doc.new_page()
        y = 72
        page.insert_text((72, 40), "PMBOK Guide - Test Document", fontsize=8, color=(0.5, 0.5, 0.5))
        page.insert_text((72, y), heading, fontsize=size, fontname="helv")
        y += 30
        for para in paragraphs:
            page.insert_textbox((72, y, 520, y + 100), para, fontsize=11, fontname="helv")
            y += 90
        page.insert_text((72, 780), f"Page {i+1}", fontsize=8, color=(0.5, 0.5, 0.5))

    doc.save(output_path)
    doc.close()
    print(f"PDF de test créé : {output_path}")


if __name__ == "__main__":
    make_sample_pdf("../data/raw/sample_pmbok.pdf")
