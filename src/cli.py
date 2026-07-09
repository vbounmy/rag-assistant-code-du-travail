import re
import sys
from collections import Counter

import chromadb
from prompt import generate_answer
from sentence_transformers import SentenceTransformer

from config import CHROMA_PERSIST_PATH, EMBEDDING_MODEL

COLLECTION_NAME = "code_du_travail"
SEUIL_CONFIANCE = 0.5
STOP_WORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "comme", "dans", "de", "des", "du", "elle", "en",
    "et", "est", "etre", "être", "for", "il", "je", "la", "le", "les", "leur", "mais", "mes", "notre",
    "nous", "on", "ou", "par", "pour", "qu", "que", "qui", "sa", "se", "ses", "sur", "ta",
    "te", "tes", "un", "une", "vos", "votre", "vous", "y", "sont", "quelle", "quelles",
    "quel", "quels", "combien", "comment", "pourquoi", "peut", "peuvent", "doit", "dans",
    "cest", "c'est", "lequel", "laquelle", "lesquels", "lesquelles",
}

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def _tokeniser(texte: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+(?:'[a-zA-ZÀ-ÿ]+)?", texte.lower())
    return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def _score_lexical(question: str, document: str) -> float:
    q_tokens = Counter(_tokeniser(question))
    d_tokens = Counter(_tokeniser(document))
    if not q_tokens:
        return 0.0
    shared = sum(min(q_tokens[token], d_tokens[token]) for token in q_tokens if token in d_tokens)
    return shared / max(1, len(q_tokens))


def _enrichir_question(question: str) -> str:
    q = question.lower()
    expansions = []
    if any(term in q for term in ["préavis", "preavis", "démission", "demission"]):
        expansions.append("préavis démission")
    if any(term in q for term in ["cdd", "durée déterminée", "duree determinee", "renouvel"]):
        expansions.append("contrat à durée déterminée renouvellement")
    if any(term in q for term in ["cdi", "durée indéterminée", "duree indeterminee"]):
        expansions.append("contrat à durée indéterminée")
    if any(term in q for term in ["congés", "conges"]):
        expansions.append("congés payés")
    if any(term in q for term in ["licenciement", "motif économique", "motif economique", "motif"]):
        expansions.append("licenciement motif économique")
    if "travail" in q and ("hebdomadaire" in q or "horaire" in q):
        expansions.append("durée légale du travail")
    if any(term in q for term in ["durée", "duree", "hebdomadaire", "mensuel", "mois"]):
        expansions.append("durée légale")
    if any(term in q for term in ["définition", "definition", "qu'est-ce", "quest-ce"]):
        expansions.append("définition article")
    return " ".join([question, *expansions])


def _normalize_article_number(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def retrieve_chunks(question, top_k=3):
    question = str(question or "").strip()
    if not question:
        return []

    model = _get_model()
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    query = _enrichir_question(question)
    embedding = model.encode(query, convert_to_numpy=True).tolist()
    resultats = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k * 12, 60),
        include=["documents", "metadatas", "distances"],
    )

    candidates: dict[str, dict[str, object]] = {}
    for document, metadata, distance in zip(
        resultats["documents"][0],
        resultats["metadatas"][0],
        resultats["distances"][0],
    ):
        article = _normalize_article_number((metadata or {}).get("numero_article"))
        if not article:
            continue
        text = str(document or "")
        lexical = _score_lexical(question, text) + _score_lexical(query, text)
        vector = 1.0 / (1.0 + max(float(distance), 1e-9))
        combined = 0.55 * vector + 0.45 * lexical
        if "licenciement" in question.lower() and "motif économique" in query:
            if "licenciement" in text or "motif économique" in text:
                combined += 0.08
        existing = candidates.get(article)
        if existing is None or combined > existing["score"]:
            candidates[article] = {
                "article": article,
                "texte": text,
                "score": combined,
            }

    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[:top_k]



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

        if not chunks:
            print("\nJe ne trouve pas cette information dans ma base de connaissances.")
            print("Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle.")
            continue

        reponse = generate_answer(question, chunks)
        afficher_reponse(reponse, chunks)


if __name__ == "__main__":
    main()