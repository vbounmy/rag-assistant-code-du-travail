# RAG Assistant - Code du travail

## Questions de réflexion

### 1. Granularité du chunking

Deux approches possibles :

- **Par article** : traçabilité parfaite (1 chunk = 1 numéro d'article), mais un article court perd son contexte thématique.
- **Par section** : garde le contexte global, mais dilue la pertinence de recherche et rend la citation floue.

**Choix retenu : approche hybride.** On indexe par article (précision de recherche et citation fiable), en enrichissant chaque chunk avec le titre de la section thématique en métadonnée et en préfixe du texte embeddé.

### 2. Traçabilité

Le numéro d'article est stocké **à la fois** dans les métadonnées (affichage fiable) et en préfixe du texte embeddé (ex: "Article L3141-1 — [texte]") pour que le LLM le voie naturellement.

Pour garantir une citation correcte : le prompt système fournit un contexte numéroté explicitement (ex: "[Source 1 - Article L1234-5]") et interdit formellement de citer un article absent du contexte fourni. Une vérification post-génération peut recouper les numéros cités avec les chunks envoyés.

### 3. Fraîcheur

Une date de constitution du corpus est stockée avec la base et affichée à chaque réponse ou en en-tête de la CLI. Le système rappelle explicitement que le droit du travail évolue et invite à vérifier la version en vigueur sur Légifrance.

### 4. Réponses conditionnelles

Le prompt pousse le LLM à donner la règle générale accompagnée de réserves explicites (taille d'entreprise, convention collective) plutôt que de deviner la situation de l'utilisateur, sans multiplier les questions de clarification.

### 5. Frontière du conseil juridique

- Question factuelle (réponse directe dans le Code) → réponse avec citation d'article.
- Question d'interprétation d'une situation personnelle → le système explique le cadre légal général mais ne tranche pas, et renvoie vers un professionnel (avocat, inspection du travail), avec l'avertissement juridique.

## Objectif du jalon 1

Le jalon 1 consiste à préparer un corpus exploitable à partir des données disponibles dans le dossier data, puis à produire un fichier JSONL prêt à être utilisé par les jalons suivants.

### Ce qui a été mis en place

- prise en charge des données locales au format Parquet dans data ;
- préparation d’un corpus structuré à partir des entrées du parquet ;
- nettoyage et normalisation du texte ;
- enrichissement des documents avec métadonnées et numéro d’article ;
- génération d’embeddings via le modèle défini dans src/config.py ;
- sortie dans le fichier data/legi_corpus.jsonl.

## Structure du projet

- src/config.py : configuration générale du projet (chemins, modèle d’embedding, clé API si nécessaire).
- src/download_legi_data.py : vérifie la présence des données locales et peut récupérer les fichiers depuis Hugging Face si nécessaire.
- src/prepare_corpus.py : lit les fichiers de données, prépare le corpus, génère les embeddings et écrit le JSONL de sortie.

## Prérequis

- Python 3.10+
- dépendances installées avec :

```bash
pip install -r requirements.txt
```

- un fichier Parquet dans data, par exemple :

```text
data/legi_code_du_travail_part_0.parquet
```

- si vous souhaitez télécharger depuis Hugging Face, définir la variable d’environnement HF_TOKEN.

## Utilisation

### 1. Vérifier ou récupérer les données

```bash
py .\src\download_legi_data.py
```

### 2. Préparer le corpus

```bash
py .\src\prepare_corpus.py
```

La sortie est générée dans :

```text
data/legi_corpus.jsonl
```

## Sortie produite

Le fichier JSONL contient un document par entrée préparée avec les champs suivants :

- id
- article_number
- title
- text
- section
- source
- source_path
- metadata
- embedding_model
- embedding

## Notes importantes

- Les données brutes et les artefacts générés ne sont pas versionnés par défaut ; voir .gitignore.
- Le jalon 1 fournit le corpus préparé. Les jalons suivants pourront s’appuyer sur ce JSONL pour l’indexation vectorielle et le RAG.

## Questions de conception retenues

### Granularité du chunking

L’approche retenue est hybride : chaque document est construit autour de l’article, tout en conservant le contexte sectionnel dans les métadonnées et dans le texte enrichi.

### Traçabilité

Le numéro d’article est conservé à la fois dans les métadonnées et dans le contenu traité afin de faciliter les citations et la vérification.

### Fraîcheur

Le système est pensé pour être compatible avec une mise à jour future du corpus, avec une date de constitution du corpus à intégrer plus tard dans l’interface et les réponses.
