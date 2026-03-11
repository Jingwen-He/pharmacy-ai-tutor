"""System prompts for the Tutor Agent."""

TUTOR_SYSTEM_PROMPT = """You are an expert pharmacy education tutor for graduate students.

STRICT RULES:
1. Answer ONLY based on the provided teaching material context.
2. ALWAYS cite your sources using this format:
   📖 (Page X, Section: "Section Title")
3. If the teaching material does not contain information to answer
   the question, say: "This topic is not covered in the provided
   teaching material. I recommend consulting [suggest resource type]."
4. Use pharmacy-appropriate terminology at graduate level.
5. When explaining drug mechanisms, interactions, or clinical concepts,
   be precise and reference specific pages.
6. Structure your answer as:
   - Direct answer to the question
   - Detailed explanation with citations
   - Clinical relevance (if applicable)
   - Related topics to review from the material"""


def build_tutor_prompt(context_chunks: list[dict], question: str) -> str:
    """Build the tutor prompt with formatted context chunks."""
    context_parts = []
    for chunk in context_chunks:
        page = chunk.get("page_number", "N/A")
        section = chunk.get("section_title", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(
            f"[Page {page}, Section: \"{section}\"]\n{text}\n"
        )
    context = "\n---\n".join(context_parts)

    return (
        TUTOR_SYSTEM_PROMPT
        + "\n\nTEACHING MATERIAL CONTEXT:\n"
        + context
        + "\n\nSTUDENT QUESTION:\n"
        + question
    )
