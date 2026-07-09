
import argparse
import json
from pathlib import Path
from typing import Any

import chromadb

from config import CHROMA_PERSIST_PATH, EMBEDDING_MODEL, ROOT_DIR

COLLECTION_NAME = "code_du_travail"
CORPUS_PATH = ROOT_DIR / "data" / "legi_corpus_vigueur.jsonl"


class ChromaIndexer:
    def __init__(
        self,
        corpus_path: Path = CORPUS_PATH,
        persist_path: Path = CHROMA_PERSIST_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.corpus_path = corpus_path.resolve()
        self.persist_path = persist_path.resolve()
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.client = chromadb.PersistentClient(path=str(self.persist_path))

    def build(self, force: bool = False) -> None:
        existing_names = [c.name for c in self.client.list_collections()]

        if self.collection_name in existing_names and not force:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            if count > 0:
                print(
                    f"Collection '{self.collection_name}' déjà présente "
                    f"({count} documents) dans {self.persist_path} — pas de réindexation."
                )
                print(f"Modèle d'embedding tracé : {collection.metadata.get('embedding_model')}")
                return

        if self.collection_name in existing_names and force:
            print(f"--force : suppression de la collection existante '{self.collection_name}'.")
            self.client.delete_collection(self.collection_name)

        print(f"Construction de la collection '{self.collection_name}'...")
        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_model_name},
        )

        documents = self._load_documents()
        if not documents:
            raise SystemExit(f"Aucun document trouvé dans {self.corpus_path}.")

        ids, embeddings, texts, metadatas = self._to_chroma_format(documents)

        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  {min(end, len(ids))}/{len(ids)} documents insérés...")

        print(f"\nIndexation terminée : {collection.count()} documents dans '{self.collection_name}'.")
        print(f"Base persistée dans : {self.persist_path}")

    def _load_documents(self) -> list[dict[str, Any]]:
        if not self.corpus_path.exists():
            raise SystemExit(f"Fichier introuvable : {self.corpus_path}")

        documents = []
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
        print(f"{len(documents)} documents chargés depuis {self.corpus_path}.")
        return documents

    def _to_chroma_format(
        self, documents: list[dict[str, Any]]
    ) -> tuple[list[str], list[list[float]], list[str], list[dict[str, Any]]]:
        ids, embeddings, texts, metadatas = [], [], [], []
        seen_ids: set[str] = set()

        for doc in documents:
            if not doc.get("text") or not doc.get("embedding"):
                continue  

            doc_id = doc["id"]
            
            if doc_id in seen_ids:
                doc_id = f"{doc_id}::{len(seen_ids)}"
            seen_ids.add(doc_id)

            ids.append(doc_id)
            embeddings.append(doc["embedding"])
            texts.append(doc["text"])
            metadatas.append(self._flatten_metadata(doc))

        return ids, embeddings, texts, metadatas

    def _flatten_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB n'accepte que des métadonnées à plat (str, int, float, bool).
        On aplatit et on nettoie les valeurs None / listes / dicts imbriqués."""
        flat: dict[str, Any] = {
            "numero_article": doc.get("article_number") or "inconnu",
            "title": doc.get("title") or "",
            "section": doc.get("section") or "inconnu",
            "source": doc.get("source") or "",
        }

        nested = doc.get("metadata", {}) or {}
        for key, value in nested.items():
            if value is None or value == "":
                continue
            if isinstance(value, (list, dict)):
                continue  
            flat[key] = value

        return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexe le corpus dans ChromaDB (jalon 2).")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH, help="Fichier JSONL source.")
    parser.add_argument("--persist-path", type=Path, default=CHROMA_PERSIST_PATH, help="Dossier de persistance Chroma.")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Nom de la collection.")
    parser.add_argument("--force", action="store_true", help="Force la réindexation même si la collection existe déjà.")
    args = parser.parse_args()

    indexer = ChromaIndexer(
        corpus_path=args.corpus,
        persist_path=args.persist_path,
        collection_name=args.collection,
    )
    indexer.build(force=args.force)


if __name__ == "__main__":
    main()