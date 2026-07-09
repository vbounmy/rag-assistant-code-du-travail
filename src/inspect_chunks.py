
import random

import chromadb

from config import CHROMA_PERSIST_PATH

COLLECTION_NAME = "code_du_travail"
SAMPLE_SIZE = 10


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"Collection '{COLLECTION_NAME}' — {total} documents.")
    print(f"Modèle d'embedding tracé : {collection.metadata.get('embedding_model')}\n")

    random.seed(42)
    offsets = random.sample(range(total), min(SAMPLE_SIZE, total))

    for offset in offsets:
        result = collection.get(limit=1, offset=offset, include=["documents", "metadatas"])
        doc_id = result["ids"][0]
        text = result["documents"][0]
        meta = result["metadatas"][0]

        print("---")
        print(f"id             : {doc_id}")
        print(f"numero_article : {meta.get('numero_article')}")
        print(f"section        : {meta.get('section')}")
        print(f"title          : {meta.get('title')}")
        print(f"texte (300c)   : {text[:300]}...")
        print()


if __name__ == "__main__":
    main()
