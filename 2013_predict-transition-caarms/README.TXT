Projet
======

Prédiction de la transition psychotique à partir de l'échelle de la CAARMS
pour 27 patients. Collab avec SHU St-Anne, J Bourgin, MO Krebs

Scripts
=======

IO.py
01_classif.py

Results
=======

Parmis les 27 11 ont fait la transition et 16 ne l'on pas faite
- Sensibilité (Taux de detection de les transitions)
72.72 % soit 8 / 11 (p = 0.03)

- Spécificité (Taux de detection de ceux qui n'ont pas transité ou 1 - Faux positifs)
87.5 % soit 14 / 16 (p = 0.01)

Nous avons un taux de bonne classification moyen de 81.4 %

Voici les items de la CAARMS qui interviennnent:
[['@4.3', -0.084408961133411356],
 ['@5.4', 0.13881360208187149],
 ['@7.4', -0.11387844064581529],
 ['@7.6', 0.0634145029598185],
 ['@7.7', 0.011629021906204358]]
Le chiffre à coté donne le poids (plus il est grand EN VALEUR ABSOLUE) plus l'item participe à la prédiction

J'ai essayé de normaliser le Pre-Morbid Adjustment scale (PAS2gr) et l'exposition au canabis (CB_EXPO). Cela dégrade considérablement les résultats. Je ne sais pas encore pourquoi et la j'ai envie de te rejoindre dans le lit...

