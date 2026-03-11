"""Orchestrator Agent: routes student input to the correct agent based on intent."""

import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import Settings


class OrchestratorAgent:
    """Routes student messages to the Tutor Agent or Quiz Agent based on intent."""

    # Keywords for quiz-related intent
    QUIZ_KEYWORDS = [
        "quiz", "test", "practice", "question me", "assess",
        "examine", "drill", "give me a question", "test me",
    ]

    # Keywords for quiz answer evaluation
    ANSWER_KEYWORDS = [
        "answer is", "my answer", "i think it's", "i think it is",
        "i choose", "i select", "option", "the answer",
        "i believe it's", "i believe it is", "i'll go with",
    ]

    # Greeting patterns
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy)\b",
        r"^(what's up|sup|yo)\b",
    ]

    def __init__(self):
        self.llm = ChatAnthropic(
            model=Settings.MODEL_NAME,
            api_key=Settings.get_api_key(),
            max_tokens=256,
            temperature=0,
        )

    def classify_intent(self, message: str) -> str:
        """Classify the student's message intent.

        Returns one of: "quiz", "answer", "question", "greeting"
        """
        message_lower = message.lower().strip()

        # Check for greetings first
        for pattern in self.GREETING_PATTERNS:
            if re.match(pattern, message_lower):
                return "greeting"

        # Check for quiz answer evaluation
        for keyword in self.ANSWER_KEYWORDS:
            if keyword in message_lower:
                return "answer"

        # Check for quiz request
        for keyword in self.QUIZ_KEYWORDS:
            if keyword in message_lower:
                return "quiz"

        # Default: treat as a content question
        return "question"

    def classify_intent_with_llm(self, message: str) -> str:
        """Use Claude to classify ambiguous intents as a fallback.

        Returns one of: "quiz", "answer", "question", "greeting"
        """
        system_prompt = """You are an intent classifier for a pharmacy education tutor system.

Classify the student's message into exactly one of these categories:
- "quiz": The student wants a practice question or quiz
- "answer": The student is answering a quiz question
- "question": The student is asking about course content
- "greeting": The student is greeting or making off-topic conversation

Respond with ONLY the category name, nothing else."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ]

        response = self.llm.invoke(messages)
        intent = response.content.strip().lower().strip('"')

        if intent in ("quiz", "answer", "question", "greeting"):
            return intent
        return "question"  # Default fallback

    def get_greeting_response(self) -> str:
        """Return a friendly greeting that redirects to study activities."""
        return (
            "Hello! 👋 I'm your Pharmacy AI Tutor. I can help you with:\n\n"
            "📚 **Ask a question** — Ask me anything about the uploaded teaching material, "
            "and I'll explain it with specific page references.\n\n"
            "📝 **Take a quiz** — Type something like \"quiz me on pharmacokinetics\" "
            "and I'll generate a practice question for you.\n\n"
            "How would you like to study today?"
        )

    def route(self, message: str) -> dict:
        """Route the student's message to the appropriate agent.

        Returns a dict with:
            - agent: "tutor", "quiz", or "greeting"
            - mode: "generate" (for quiz), "evaluate" (for answers), or None
            - message: the original message
        """
        intent = self.classify_intent(message)

        if intent == "greeting":
            return {
                "agent": "greeting",
                "mode": None,
                "message": message,
            }
        elif intent == "quiz":
            return {
                "agent": "quiz",
                "mode": "generate",
                "message": message,
            }
        elif intent == "answer":
            return {
                "agent": "quiz",
                "mode": "evaluate",
                "message": message,
            }
        else:
            return {
                "agent": "tutor",
                "mode": None,
                "message": message,
            }
