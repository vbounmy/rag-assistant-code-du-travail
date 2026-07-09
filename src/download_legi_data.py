import argparse
from pathlib import Path

from config import DATA_DIR, ROOT_DIR, HF_TOKEN

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:  # pragma: no cover - dépendance optionnelle si les données sont déjà locales
    hf_hub_download = None
    list_repo_files = None


class HuggingFaceDatasetDownloader:
    def __init__(self, repo_id: str = "AgentPublic/legi", repo_type: str = "dataset", prefix: str = "data/", data_dir: Path | None = None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.prefix = prefix
        self.data_dir = (data_dir or DATA_DIR).resolve()
        self.repo_root = ROOT_DIR

    def download_all(self) -> int:
        local_files = self._collect_local_files()
        if local_files:
            print(f"{len(local_files)} fichier(s) local déjà présent(s) dans {self.data_dir}.")
            return len(local_files)

        if hf_hub_download is None or list_repo_files is None:
            raise SystemExit(
                "huggingface_hub est requis pour télécharger des données depuis Hugging Face. "
                "Si vous avez déjà des fichiers dans data/, lancez simplement le script de préparation."
            )

        file_paths = self._list_data_files()
        if not file_paths:
            raise SystemExit(f"Aucun fichier trouvé dans {self.repo_id}/{self.prefix}.")

        print(f"{len(file_paths)} fichier(s) trouvé(s) dans {self.repo_id}/{self.prefix}.")
        downloaded = 0
        for file_path in file_paths:
            if self._download_file(file_path):
                downloaded += 1

        print(f"Téléchargement terminé. Fichiers disponibles dans {self.data_dir}.")
        return downloaded

    def _collect_local_files(self) -> list[Path]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".parquet", ".jsonl", ".json", ".csv", ".xml", ".gz"}
        )

    def _list_data_files(self) -> list[str]:
        repo_files = list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
        return sorted(path for path in repo_files if path.startswith(self.prefix))

    def _download_file(self, file_path: str) -> bool:
        local_path = self.repo_root / file_path
        if local_path.exists():
            print(f"Déjà présent : {local_path}")
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement de {file_path} -> {local_path}...")
        hf_hub_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            filename=file_path,
            local_dir=str(self.repo_root),
            local_dir_use_symlinks=False,
            token=HF_TOKEN or None,
        )
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge ou vérifie la présence des données LEGI locales.")
    parser.add_argument("--repo-id", default="AgentPublic/legi", help="Identifiant du dataset Hugging Face.")
    parser.add_argument("--repo-type", default="dataset", help="Type du dépôt Hugging Face.")
    parser.add_argument("--prefix", default="data/", help="Préfixe des fichiers à télécharger.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Répertoire de données local.")
    args = parser.parse_args()

    downloader = HuggingFaceDatasetDownloader(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        prefix=args.prefix,
        data_dir=args.data_dir,
    )
    downloader.download_all()


if __name__ == "__main__":
    main()
