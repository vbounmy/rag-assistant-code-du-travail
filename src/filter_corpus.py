"""
Filtre legi_corpus.jsonl pour ne garder que les articles réellement en vigueur.

Problème identifié au jalon 3 : le corpus contient des versions obsolètes
(status MODIFIE), abrogées (ABROGE / ABROGE_DIFF), transférées (TRANSFERE)
et jamais entrées en vigueur (MODIFIE_MORT_NE), qui polluent le retrieval
et peuvent faire remonter du droit obsolète (ex: ancien article L321-2
au lieu de l'article en vigueur L1233-3 sur le licenciement économique).

Ne nécessite pas de recalculer les embeddings : on filtre le JSONL existant.
"""

import json

from config import ROOT_DIR

INPUT_PATH = ROOT_DIR / "data" / "legi_corpus.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "legi_corpus_vigueur.jsonl"

# Statuts à conserver : uniquement le droit actuellement applicable
STATUTS_VALIDES = {"VIGUEUR", "VIGUEUR_DIFF"}


def main():
    total = 0
    conserves = 0
    compte_par_statut: dict[str, int] = {}

    with INPUT_PATH.open("r", encoding="utf-8") as source, OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as sortie:
        for line in source:
            line = line.strip()
            if not line:
                continue
            total += 1
            doc = json.loads(line)
            statut = doc.get("metadata", {}).get("status") or "inconnu"
            compte_par_statut[statut] = compte_par_statut.get(statut, 0) + 1

            if statut in STATUTS_VALIDES:
                sortie.write(json.dumps(doc, ensure_ascii=False) + "\n")
                conserves += 1

    print(f"Total lu       : {total}")
    print(f"Conservés      : {conserves} ({conserves / total * 100:.1f}%)")
    print(f"Exclus         : {total - conserves}")
    print(f"\nRépartition par statut :")
    for statut, count in sorted(compte_par_statut.items(), key=lambda x: -x[1]):
        marque = "conservé" if statut in STATUTS_VALIDES else "exclu"
        print(f"  {statut:20s} {count:6d}  ({marque})")

    print(f"\nFichier filtré écrit dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
