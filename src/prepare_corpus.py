import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from config import EMBEDDING_MODEL, ROOT_DIR, HF_TOKEN

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from sentence_transformers import SentenceTransformer


class LegiCorpusPreparer:
    def __init__(
        self,
        data_dir: Path = ROOT_DIR / "data",
        output_path: Path = ROOT_DIR / "data" / "legi_corpus.jsonl",
        embedding_model: str = EMBEDDING_MODEL,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.data_dir = data_dir.resolve()
        self.output_path = output_path.resolve()
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def prepare(self) -> None:
        random.seed(self.seed)
        parquet_files = self._collect_parquet_files()
        if not parquet_files:
            raise SystemExit(f"Aucun fichier Parquet trouvé dans {self.data_dir}. Placez les fichiers LEGI ici.")

        print(f"{len(parquet_files)} fichier(s) Parquet trouvé(s) dans {self.data_dir}.")
        rows: list[dict[str, Any]] = []
        for path in parquet_files:
            rows.extend(self._parse_parquet(path))

        if not rows:
            raise SystemExit("Aucune entrée extraite. Vérifiez le contenu du fichier Parquet.")

        texts = [self._select_text(row) for row in rows]
        embeddings = self._embed_texts(texts)

        documents = []
        for row, embedding in zip(rows, embeddings):
            document = self._build_document(row, embedding)
            documents.append(document)

        self._save_jsonl(documents)
        print(f"Corpus enregistré dans {self.output_path} ({len(documents)} documents).\n")
        self._display_samples(documents)

    def _collect_parquet_files(self) -> list[Path]:
        return sorted(self.data_dir.rglob("*.parquet"))

    def _parse_parquet(self, path: Path) -> list[dict[str, Any]]:
        print(f"Lecture de {path}...")
        dataframe = pd.read_parquet(path)
        rows = dataframe.to_dict(orient="records")
        return [self._normalize_row(row, path) for row in rows]

    def _normalize_row(self, row: dict[str, Any], path: Path) -> dict[str, Any]:
        normalized = {}
        for key, value in row.items():
            if pd.isna(value):
                normalized[key] = None
            elif isinstance(value, (list, dict)):
                normalized[key] = value
            else:
                normalized[key] = str(value) if not isinstance(value, (int, float, bool)) else value
        normalized["__source_path__"] = str(path)
        return normalized

    def _select_text(self, row: dict[str, Any]) -> str:
        for column_name in ("chunk_text", "text", "full_text", "content"):
            value = self._clean_text(row.get(column_name))
            if value:
                return value
        return ""

    def _clean_text(self, text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, float) and pd.isna(text):
            return ""
        return str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = SentenceTransformer(self.embedding_model_name, device=self.device)
        embeddings = model.encode(texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()

    def _build_document(self, row: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
        title = self._clean_text(row.get("title") or row.get("full_title") or row.get("number"))
        article_number = self._clean_text(row.get("number") or row.get("doc_id") or row.get("chunk_id") or "unknown")
        section = self._clean_text(row.get("category") or row.get("nature") or row.get("ministry") or "inconnu")
        text = self._select_text(row)
        source_name = self._clean_text(row.get("doc_id") or row.get("title") or row.get("__source_path__"))
        chunk_id = self._clean_text(row.get("chunk_id") or row.get("doc_id") or row.get("id"))

        return {
            "id": chunk_id or f"{article_number}:{len(embedding)}",
            "article_number": article_number,
            "title": title,
            "text": text,
            "section": section,
            "source": source_name,
            "source_path": self._clean_text(row.get("__source_path__")),
            "metadata": {
                "chunk_id": self._clean_text(row.get("chunk_id")),
                "doc_id": self._clean_text(row.get("doc_id")),
                "chunk_index": self._clean_text(row.get("chunk_index")),
                "nature": self._clean_text(row.get("nature")),
                "category": self._clean_text(row.get("category")),
                "ministry": self._clean_text(row.get("ministry")),
                "status": self._clean_text(row.get("status")),
                "number": self._clean_text(row.get("number")),
                "start_date": self._clean_text(row.get("start_date")),
                "end_date": self._clean_text(row.get("end_date")),
                "links": row.get("links"),
            },
            "embedding_model": self.embedding_model_name,
            "embedding": embedding,
        }

    def _save_jsonl(self, documents: list[dict]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    def _display_samples(self, documents: list[dict], sample_size: int = 10) -> None:
        sample = documents if len(documents) <= sample_size else random.sample(documents, sample_size)
        print(f"Affichage de {len(sample)} documents au hasard:\n")
        for item in sample:
            print("---")
            print(f"id: {item['id']}")
            print(f"article_number: {item['article_number']}")
            print(f"section: {item['section']}")
            print(f"source: {item['source']}")
            print(f"texte: {item['text'][:400]}...")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare un corpus LEGI en JSONL pour la recherche vectorielle.")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data", help="Répertoire contenant les fichiers Parquet LEGI.")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data" / "legi_corpus.jsonl", help="Fichier JSONL de sortie.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Modèle d'embedding à utiliser.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de lot pour l'encodage.")
    parser.add_argument("--seed", type=int, default=42, help="Graine pour l'échantillonnage aléatoire.")
    args = parser.parse_args()

    preparer = LegiCorpusPreparer(
        data_dir=args.data_dir,
        output_path=args.output,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    preparer.prepare()


if __name__ == "__main__":
    main()
