from openai import OpenAI

from .config import OPENROUTER_BASE_URL, OPENROUTER_MODEL
from .security import get_openrouter_key


class MissingAPIKeyError(RuntimeError):
    pass


class RAGService:
    def __init__(self) -> None:
        api_key = get_openrouter_key()

        if not api_key:
            raise MissingAPIKeyError(
                "Enter an OpenRouter API key in Settings before using AI features."
            )

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=45.0,
            max_retries=2,
        )

    @staticmethod
    def _context(results: list[dict]) -> str:
        parts = []

        for number, result in enumerate(results, start=1):
            description = result.get("description", "")[:1600]

            parts.append(
                f"[{number}] {result.get('title', '')}\n"
                f"{description}"
            )

        return "\n\n".join(parts)

    def generate(
        self,
        query: str,
        results: list[dict],
        mode: str,
    ) -> str:
        context = self._context(results)

        instructions = {
            "Question": (
                "Answer the question directly and concisely. "
                "Use the supplied movie documents as evidence."
            ),
            "Summary": (
                "Synthesize the results into a concise, information-dense "
                "3-5 sentence summary."
            ),
            "Citations": (
                "Answer comprehensively and cite factual statements using "
                "[1], [2], etc. corresponding to the supplied documents."
            ),
        }

        task = instructions.get(mode, instructions["Question"])

        prompt = f"""You are the assistant inside Webflyx RAG Studio.

Only use the retrieved documents below as factual grounding.
If the documents do not contain enough information, explicitly say so.
Do not invent facts, cast members, dates, ratings, or plot details.

Task:
{task}

User query:
{query}

Retrieved documents:
{context}

Answer:"""

        response = self.client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return (
            response.choices[0].message.content
            or "No response was returned."
        ).strip()
