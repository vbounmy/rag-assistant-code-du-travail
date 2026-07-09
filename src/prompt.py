import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le Code du travail français.

RÔLE :
Tu réponds aux questions des utilisateurs UNIQUEMENT à partir des extraits du Code du travail fournis ci-dessous dans le contexte. Tu ne dois jamais utiliser tes connaissances générales sur le droit du travail.

RÈGLES STRICTES :
1. Chaque affirmation de ta réponse doit être appuyée par au moins un article présent dans le contexte fourni.
2. Tu dois systématiquement citer le numéro de l'article sur lequel tu t'appuies (ex: "selon l'article L3141-1...").
3. Il est INTERDIT d'inventer un numéro d'article ou de citer un article qui n'apparaît pas dans le contexte ci-dessous.
4. Si le contexte fourni ne contient pas l'information nécessaire pour répondre à la question, tu dois répondre exactement : "Je ne trouve pas cette information dans ma base de connaissances." Ne tente jamais de deviner ou de compléter avec tes connaissances générales.
5. Si la question dépend de facteurs non précisés (taille de l'entreprise, convention collective, ancienneté...), donne la règle générale trouvée dans le contexte, en précisant explicitement que cela peut varier selon la situation.
6. Si la question demande une interprétation d'une situation personnelle (ex: "mon licenciement est-il abusif ?"), explique le cadre légal général à partir du contexte, mais ne tranche jamais le cas personnel de l'utilisateur.

CONTEXTE (articles du Code du travail) :
{context}

Termine TOUJOURS ta réponse par cette phrase, mot pour mot :
"Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle."
"""


def build_context(chunks):
    """
    chunks : liste de dicts avec au moins 'article' et 'texte' (et 'section' en option)
    Retournés par la fonction de retrieval (jalon 3) de ta collègue.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i} - Article {chunk['article']}]\n{chunk['texte']}"
        )
    return "\n\n".join(parts)


def generate_answer(question, chunks, model="llama-3.3-70b-versatile"):
    """
    question : la question posée par l'utilisateur (str)
    chunks   : les chunks pertinents renvoyés par le retrieval (jalon 3)
    """
    context = build_context(chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


# Test rapide en isolé, avec des chunks factices
# (à supprimer ou mettre sous if __name__ == "__main__" une fois le vrai retrieval branché)
if __name__ == "__main__":
    fake_chunks = [
        {
            "article": "L3141-3",
            "texte": "Le salarié a droit à un congé de deux jours et demi ouvrables par mois de travail effectif chez le même employeur.",
        }
    ]
    question = "Combien de jours de congés payés par mois ?"
    print(generate_answer(question, fake_chunks))