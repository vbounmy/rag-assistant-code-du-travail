import json
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PERSIST_PATH, EMBEDDING_MODEL, ROOT_DIR

COLLECTION_NAME = "code_du_travail"
QUESTIONS_FILE = ROOT_DIR / "data" / "questions_test.json"  
TOP_K = 5


def charger_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rechercher(collection, model, question, top_k=TOP_K):
    embedding_question = model.encode(question).tolist()
    return collection.query(query_embeddings=[embedding_question], n_results=top_k)


def extraire_articles(resultats):
    return [meta.get("numero_article") for meta in resultats["metadatas"][0]]


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    questions_test = charger_questions(QUESTIONS_FILE)
    print(f"Validation du retrieval — top-{TOP_K}\n" + "=" * 60)
    nb_ok = 0

    for item in questions_test:
        resultats = rechercher(collection, model, item["question"])
        articles_trouves = extraire_articles(resultats)
        distances = resultats["distances"][0]

        ok = item["article_attendu"] in articles_trouves
        nb_ok += ok

        statut = "OK" if ok else "ECHEC"
        print(f"\n[{statut}] [{item['theme']}] {item['question']}")
        print(f"   Attendu   : {item['article_attendu']}")
        print(f"   Trouvés   : {articles_trouves}")
        print(f"   Distances : {[round(d, 3) for d in distances]}")

    print("\n" + "=" * 60)
    print(f"Score : {nb_ok}/{len(questions_test)}")


if __name__ == "__main__":
    main()
