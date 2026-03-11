"""Quiz Agent: generates practice questions and evaluates student answers."""

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import Settings
from prompts.quiz_prompt import build_quiz_generation_prompt, build_quiz_evaluation_prompt


class QuizAgent:
    """Generates quiz questions from teaching material and evaluates student answers."""

    def __init__(self):
        self.llm = ChatAnthropic(
            model=Settings.MODEL_NAME,
            api_key=Settings.get_api_key(),
            max_tokens=2048,
            temperature=0.7,  # Higher temperature for question variety
        )

    def generate_question(
        self, topic: str, context_chunks: list[dict]
    ) -> dict | None:
        """Generate a practice question based on retrieved context.

        Args:
            topic: The topic or chapter the student wants to be quizzed on.
            context_chunks: Retrieved chunks from the Retrieval Agent.

        Returns:
            A dict with keys: question, type, options, correct_answer,
            explanation, source, difficulty. Returns None on failure.
        """
        if not context_chunks:
            return None

        system_prompt = build_quiz_generation_prompt(context_chunks, topic)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Generate a practice question about: {topic}"
            ),
        ]

        response = self.llm.invoke(messages)
        return self._parse_quiz_response(response.content)

    def evaluate_answer(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        context_chunks: list[dict],
    ) -> str:
        """Evaluate a student's answer against the correct answer.

        Args:
            question: The quiz question that was asked.
            correct_answer: The correct answer from quiz generation.
            student_answer: The student's submitted answer.
            context_chunks: Retrieved context for citation support.

        Returns:
            Evaluation feedback string with assessment, corrections, and study tips.
        """
        system_prompt = build_quiz_evaluation_prompt(
            question=question,
            correct_answer=correct_answer,
            student_answer=student_answer,
            context_chunks=context_chunks,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Evaluate this answer: {student_answer}"),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _parse_quiz_response(self, response_text: str) -> dict | None:
        """Parse the LLM's quiz question response into a structured dict."""
        try:
            # Try to extract JSON from the response
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            quiz_data = json.loads(cleaned)

            # Validate required fields
            required_fields = [
                "question", "type", "correct_answer", "explanation", "source", "difficulty"
            ]
            for field in required_fields:
                if field not in quiz_data:
                    return None

            # Ensure options is a list or None
            if "options" not in quiz_data:
                quiz_data["options"] = None

            return quiz_data

        except (json.JSONDecodeError, KeyError):
            # If JSON parsing fails, try to extract from the text
            return self._fallback_parse(response_text)

    def _fallback_parse(self, text: str) -> dict | None:
        """Attempt to extract quiz data from non-JSON response."""
        try:
            # Try to find JSON embedded in the text
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None
