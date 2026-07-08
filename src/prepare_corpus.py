import argparse
import gzip
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTICLE_TAGS = {"article", "articletexte", "art", "articl"}
SECTION_TAGS = {"section", "soussection", "soustion", "chapitre", "titre", "rubrique", "partie", "division", "book", "fascicule", "sels"}
TITLE_TAGS = {"titre", "denomination", "libelle", "designation", "intitule", "label", "num"}

NAMESPACE_RE = re.compile(r"\{.*\}")
CLEANUP_RE = re.compile(r"\s+")
REFERENCE_RE = re.compile(r"\s*\[[0-9,\s]+\]")
PAREN_RE = re.compile(r"\s*\([^)]*(?:voir|cf\.|cf|article|source)[^)]*\)", flags=re.I)


class LegiCorpusPreparer:
    def __init__(self, data_dir: Path = ROOT_DIR / "data", output_path: Path = ROOT_DIR / "data" / "legi_corpus.jsonl", seed: int = 42):
        self.data_dir = data_dir.resolve()
        self.output_path = output_path.resolve()
        self.seed = seed

    def prepare(self) -> None:
        random.seed(self.seed)
        xml_files = self._collect_xml_files()
        if not xml_files:
            raise SystemExit(f"Aucun fichier XML trouvé dans {self.data_dir}. Placez les fichiers LEGI ici.")

        print(f"{len(xml_files)} fichier(s) XML trouvé(s) dans {self.data_dir}.")
        documents = [doc for path in xml_files for doc in self._parse_file(path)]
        if not documents:
            raise SystemExit("Aucune entrée extraite. Vérifiez la structure XML et adaptez le parser si nécessaire.")

        self._save_jsonl(documents)
        print(f"Corpus enregistré dans {self.output_path} ({len(documents)} documents).\n")
        self._display_samples(documents)

    def _collect_xml_files(self) -> list[Path]:
        files = sorted(self.data_dir.rglob("*.xml"))
        files.extend(sorted(self.data_dir.rglob("*.xml.gz")))
        return files

    def _parse_file(self, path: Path) -> list[dict]:
        try:
            with self._open_xml(path) as stream:
                tree = ET.parse(stream)
        except ET.ParseError as exc:
            print(f"WARN: impossible de parser {path}: {exc}")
            return []

        root = tree.getroot()
        parent_map = self._build_parent_map(root)
        articles = [elem for elem in root.iter() if self._is_article_element(elem)]

        documents = []
        for article in articles:
            article_number = self._extract_article_number(article) or "unknown"
            title = self._find_first_subtext(article, TITLE_TAGS) or ""
            raw_text = self._get_text(article)
            cleaned_text = self._clean_text(raw_text)
            section = self._find_section_labels(article, parent_map)
            text = f"{title}. {cleaned_text}".strip() if title else cleaned_text
            doc_id = f"{path.name}:{article_number}" if article_number != "unknown" else f"{path.name}:{len(documents)+1}"

            documents.append({
                "id": doc_id,
                "article_number": article_number,
                "title": title,
                "text": text,
                "section": section,
                "source": path.name,
                "source_path": str(path),
            })

        return documents

    def _open_xml(self, path: Path):
        return gzip.open(path, mode="rb") if path.suffix.lower() == ".gz" or path.name.lower().endswith(".xml.gz") else open(path, mode="rb")

    @staticmethod
    def _get_text(element: ET.Element) -> str:
        return " ".join(text.strip() for text in element.itertext() if text and text.strip())

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = REFERENCE_RE.sub("", text)
        text = PAREN_RE.sub("", text)
        return CLEANUP_RE.sub(" ", text).strip()

    @staticmethod
    def _local_name(tag: str) -> str:
        return NAMESPACE_RE.sub("", tag) if isinstance(tag, str) else ""

    def _build_parent_map(self, root: ET.Element) -> dict[ET.Element, ET.Element]:
        return {child: parent for parent in root.iter() for child in parent}

    def _find_first_subtext(self, element: ET.Element, tag_names: set[str]) -> str | None:
        for child in element:
            if self._local_name(child.tag).lower() in tag_names:
                value = self._get_text(child)
                if value:
                    return value
        for child in element:
            value = self._find_first_subtext(child, tag_names)
            if value:
                return value
        return None

    def _find_section_labels(self, article: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> str:
        labels = []
        current = parent_map.get(article)
        while current is not None:
            tag = self._local_name(current.tag).lower()
            if tag in SECTION_TAGS:
                title = self._find_first_subtext(current, TITLE_TAGS)
                if title and title not in labels:
                    labels.append(title)
            current = parent_map.get(current)
        return " > ".join(reversed(labels)) if labels else "inconnu"

    def _extract_article_number(self, article: ET.Element) -> str | None:
        num = self._find_first_subtext(article, {"num", "numero", "article"})
        if num:
            return num.strip()
        for attr in ("id", "xml:id"):
            value = article.attrib.get(attr)
            if value:
                return value.strip()
        return None

    def _is_article_element(self, element: ET.Element) -> bool:
        return self._local_name(element.tag).lower() in ARTICLE_TAGS

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
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data", help="Répertoire contenant les fichiers XML LEGI.")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data" / "legi_corpus.jsonl", help="Fichier JSONL de sortie.")
    parser.add_argument("--seed", type=int, default=42, help="Graine pour l'échantillonnage aléatoire.")
    args = parser.parse_args()

    preparer = LegiCorpusPreparer(data_dir=args.data_dir, output_path=args.output, seed=args.seed)
    preparer.prepare()


if __name__ == "__main__":
    main()
