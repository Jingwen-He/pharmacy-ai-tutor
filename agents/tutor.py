"""Tutor Agent: answers student questions using retrieved teaching material with citations."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import Settings
from prompts.tutor_prompt import build_tutor_prompt


class TutorAgent:
    """Answers pharmacy questions using ONLY the retrieved teaching material, with citations."""

    def __init__(self):
        self.llm = ChatAnthropic(
            model=Settings.MODEL_NAME,
            api_key=Settings.get_api_key(),
            max_tokens=2048,
            temperature=0.3,
        )

    def answer(
        self,
        question: str,
        context_chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        """Generate a cited answer to the student's question.

        Args:
            question: The student's question.
            context_chunks: Retrieved chunks from the Retrieval Agent.
            chat_history: Previous conversation messages for follow-up context.

        Returns:
            The tutor's answer with citations.
        """
        if not context_chunks:
            return (
                "I couldn't find relevant information in the teaching material "
                "for your question. Could you rephrase it or ask about a specific "
                "topic covered in the uploaded material?"
            )

        system_prompt = build_tutor_prompt(context_chunks, question)

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history for follow-up support
        if chat_history:
            for msg in chat_history[-6:]:  # Last 6 messages for context
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        return response.content
