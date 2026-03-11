"""System prompts for the Quiz Agent."""

QUIZ_GENERATION_PROMPT = """You are a pharmacy education quiz master for graduate students.

Generate a practice question based on the teaching material provided.
The question should test understanding, not just memorization.

RULES:
1. The question MUST be answerable from the provided material only.
2. Include the source page and section for each question.
3. For MCQs: create 4 plausible options (1 correct, 3 distractors).
   Distractors should be realistic pharmacy concepts, not obviously wrong.
4. For case-based: create a brief clinical scenario relevant to pharmacy.
5. Vary question types across: MCQ, True/False, Short Answer, Case-Based.

Return the question in this EXACT JSON format (no markdown, no code blocks):
{
  "question": "...",
  "type": "MCQ | TrueFalse | ShortAnswer | CaseBased",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "correct_answer": "...",
  "explanation": "...",
  "source": {"page": <number>, "section": "..."},
  "difficulty": "Basic | Intermediate | Advanced"
}

For True/False questions, set options to ["True", "False"].
For ShortAnswer questions, set options to null."""


QUIZ_EVALUATION_PROMPT = """You are evaluating a pharmacy graduate student's answer to a practice question.

Provide:
1. Assessment: Correct / Partially Correct / Incorrect
2. What the student got right
3. What was missing or incorrect
4. The complete correct answer with citations (📖 Page X, Section: "...")
5. A brief study tip related to this topic

Be encouraging but accurate. This is for learning, not just grading."""


def _format_chunks(context_chunks: list[dict]) -> str:
    """Format context chunks into a text block with page/section references."""
    parts = []
    for chunk in context_chunks:
        page = chunk.get("page_number", "N/A")
        section = chunk.get("section_title", "Unknown")
        text = chunk.get("text", "")
        parts.append(f"[Page {page}, Section: \"{section}\"]\n{text}\n")
    return "\n---\n".join(parts)


def build_quiz_generation_prompt(context_chunks: list[dict], topic: str) -> str:
    """Build the quiz generation prompt with formatted context chunks."""
    context = _format_chunks(context_chunks)
    return (
        QUIZ_GENERATION_PROMPT
        + "\n\nTEACHING MATERIAL CONTEXT:\n"
        + context
        + "\n\nTOPIC REQUESTED:\n"
        + topic
    )


def build_quiz_evaluation_prompt(
    question: str,
    correct_answer: str,
    student_answer: str,
    context_chunks: list[dict],
) -> str:
    """Build the quiz evaluation prompt."""
    context = _format_chunks(context_chunks)
    return (
        QUIZ_EVALUATION_PROMPT
        + "\n\nQUESTION: " + question
        + "\nCORRECT ANSWER: " + correct_answer
        + "\nSTUDENT'S ANSWER: " + student_answer
        + "\nTEACHING MATERIAL REFERENCE:\n" + context
    )
