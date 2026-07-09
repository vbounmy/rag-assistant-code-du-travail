# rag-assistant-code-du-travail 
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
- Question d'interprétation d'une situation personnelle → le système explique le cadre légal général mais ne tranche pas, et renvoie vers un professionnel (avocat, inspection du travail), avec l'avertissement juridique.s