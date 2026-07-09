questions_test = [
    ("Quelle est la durée légale du préavis de démission pour un CDI ?", "L1237-1"),
    ("Combien de jours de congés payés acquiert-on par mois ?", "L3141-3"),
    ("Combien de fois peut-on renouveler un CDD ?", "L1243-13"),
    ("Quelle est la durée légale hebdomadaire du travail ?", "L3121-27"),
    ("Qu'est-ce que le licenciement pour motif économique ?", "L1233-3"),
]

for question, article_attendu in questions_test:
    resultats = rechercher(question, top_k=5)
    articles_trouves = [r["numero_article"] for r in resultats]
    ok = article_attendu in articles_trouves
    print(f"{question} → {'OK' if ok else 'ÉCHEC'} (attendu: {article_attendu}, trouvé: {articles_trouves})")