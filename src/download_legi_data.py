from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub est requis pour ce script. Installez-le avec : pip install huggingface_hub"
    ) from exc


class HuggingFaceDatasetDownloader:
    def __init__(self, repo_id: str = "AgentPublic/legi", repo_type: str = "dataset", prefix: str = "data/"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.prefix = prefix
        self.repo_root = Path(__file__).resolve().parent.parent

    def download_all(self) -> int:
        file_paths = self._list_data_files()
        if not file_paths:
            raise SystemExit(f"Aucun fichier trouvé dans {self.repo_id}/{self.prefix}.")

        print(f"{len(file_paths)} fichier(s) trouvé(s) dans {self.repo_id}/{self.prefix}.")
        downloaded = 0
        for file_path in file_paths:
            if self._download_file(file_path):
                downloaded += 1

        print(f"Téléchargement terminé. Fichiers disponibles dans {self.repo_root / self.prefix}.")
        return downloaded

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
            local_dir=self.repo_root,
            local_dir_use_symlinks=False,
        )
        return True


def main() -> None:
    downloader = HuggingFaceDatasetDownloader()
    downloader.download_all()


if __name__ == "__main__":
    main()
