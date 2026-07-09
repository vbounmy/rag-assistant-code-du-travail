import sys
from prompt import generate_answer

from retrieval_eval import retrieve_chunks


SEUIL_CONFIANCE = 0.5  


def retrieve_chunks_fake(question, top_k=3):
    """
    Fonction temporaire, à supprimer une fois le vrai retrieval branché.
    Retourne des chunks factices avec un score pour tester la CLI en isolé.
    """
    return [
        {
            "article": "L3141-3",
            "texte": "Le salarié a droit à un congé de deux jours et demi ouvrables par mois de travail effectif chez le même employeur.",
            "score": 0.87,
        }
    ]


def verifier_confiance(chunks):
    """
    chunks : liste de dicts, chacun devra contenir une clé 'score'
    une fois le retrieval de ta collègue branché (score de similarité, 0 à 1,
    plus haut = meilleure correspondance).
    Retourne True si la confiance est suffisante, False sinon.
    """
    if not chunks:
        return False
    meilleur_score = max(chunk.get("score", 1.0) for chunk in chunks)
    return meilleur_score >= SEUIL_CONFIANCE


def afficher_reponse(reponse, chunks):
    print("\n" + "=" * 60)
    print("RÉPONSE :\n")
    print(reponse)
    print("\n" + "-" * 60)
    print("Articles source(s) utilisé(s) :")
    for chunk in chunks:
        score = chunk.get("score")
        if score is not None:
            print(f"  - Article {chunk['article']} (confiance : {score:.2f})")
        else:
            print(f"  - Article {chunk['article']}")
    if not verifier_confiance(chunks):
        print("\n⚠️  Confiance faible : cette réponse pourrait être imprécise ou hors sujet.")
    print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("Assistant Code du travail — RAG")
    print("Tapez votre question, ou 'quit' / 'exit' pour quitter.")
    print("=" * 60)

    while True:
        question = input("\nVotre question : ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("À bientôt !")
            sys.exit(0)

        if not question:
            print("Merci de saisir une question.")
            continue

        chunks = retrieve_chunks(question)

        chunks = retrieve_chunks_fake(question)

        if not chunks:
            print("\nJe ne trouve pas cette information dans ma base de connaissances.")
            print("Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle.")
            continue

        reponse = generate_answer(question, chunks)
        afficher_reponse(reponse, chunks)


if __name__ == "__main__":
    main()