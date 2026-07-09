"""
Jalon 3 — Retrieval
Contient :
- retrieve_chunks(question, top_k) : fonction réutilisable par cli.py (jalon 5)
- Le script d'évaluation du retrieval sur le jeu de questions de test
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PERSIST_PATH, EMBEDDING_MODEL, ROOT_DIR

COLLECTION_NAME = "code_du_travail"
QUESTIONS_FILE = ROOT_DIR / "data" / "questions_test.json"
TOP_K = 5

# Chargés une seule fois, réutilisés à chaque appel de retrieve_chunks
_collection = None
_model = None


def _get_collection_and_model():
    global _collection, _model
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _collection, _model


def distance_vers_score(distance: float) -> float:
    """Convertit une distance ChromaDB (0 = identique, plus grand = plus différent)
    en score de confiance entre 0 et 1 (plus grand = meilleure correspondance),
    format attendu par verifier_confiance() dans cli.py."""
    return 1 / (1 + distance)


def retrieve_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    """Fonction attendue par cli.py (jalon 5) et prompt.py (jalon 4).
    Retourne une liste de dicts : article, texte, score, section."""
    collection, model = _get_collection_and_model()

    embedding_question = model.encode(question).tolist()
    resultats = collection.query(query_embeddings=[embedding_question], n_results=top_k)

    chunks = []
    for doc_id, texte, meta, distance in zip(
        resultats["ids"][0],
        resultats["documents"][0],
        resultats["metadatas"][0],
        resultats["distances"][0],
    ):
        chunks.append(
            {
                "article": meta.get("numero_article", "inconnu"),
                "texte": texte,
                "section": meta.get("section", ""),
                "score": distance_vers_score(distance),
                "distance": distance,
            }
        )
    return chunks


# --- Script d'évaluation (jalon 3, contrôle qualité) ---


def charger_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    questions_test = charger_questions(QUESTIONS_FILE)
    print(f"Validation du retrieval — top-{TOP_K}\n" + "=" * 60)
    nb_ok = 0

    for item in questions_test:
        chunks = retrieve_chunks(item["question"], top_k=TOP_K)
        articles_trouves = [c["article"] for c in chunks]

        ok = item["article_attendu"] in articles_trouves
        nb_ok += ok

        statut = "OK" if ok else "ECHEC"
        print(f"\n[{statut}] [{item['theme']}] {item['question']}")
        print(f"   Attendu   : {item['article_attendu']}")
        print(f"   Trouvés   : {articles_trouves}")
        print(f"   Scores    : {[round(c['score'], 3) for c in chunks]}")

    print("\n" + "=" * 60)
    print(f"Score : {nb_ok}/{len(questions_test)}")


if __name__ == "__main__":
    main()
