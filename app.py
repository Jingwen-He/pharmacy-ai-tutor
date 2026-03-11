"""Streamlit frontend for the AI Tutor multi-agent system."""

import os
import tempfile

import streamlit as st

from core.config import Settings
from agents.orchestrator import OrchestratorAgent
from agents.retrieval import RetrievalAgent
from agents.tutor import TutorAgent
from agents.quiz import QuizAgent


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pharmacy AI Tutor",
    page_icon="💊",
    layout="wide",
)

# ── Session state initialization ─────────────────────────────────────────────


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "pdf_processed": False,
        "chunk_count": 0,
        "sections": [],
        "current_quiz": None,  # Stores the active quiz question dict
        "mode": "Q&A Mode",
        "agents_initialized": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ── Initialize agents (cached) ──────────────────────────────────────────────


@st.cache_resource
def get_retrieval_agent():
    return RetrievalAgent()


@st.cache_resource
def get_orchestrator():
    return OrchestratorAgent()


@st.cache_resource
def get_tutor_agent():
    return TutorAgent()


@st.cache_resource
def get_quiz_agent():
    return QuizAgent()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💊 Pharmacy AI Tutor")
    st.markdown("---")

    # API key validation
    if not Settings.validate():
        st.warning("⚠️ Set your `ANTHROPIC_API_KEY` in the `.env` file to get started.")

    # PDF upload
    st.subheader("📄 Upload Teaching Material")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload your pharmacy course PDF to get started.",
    )

    if uploaded_file is not None and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                retrieval_agent = get_retrieval_agent()

                # Clear previous data
                retrieval_agent.clear()

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # Process the PDF
                chunk_count = retrieval_agent.ingest_pdf(tmp_path)

                # Clean up temp file
                os.unlink(tmp_path)

                # Update session state
                st.session_state.pdf_processed = True
                st.session_state.chunk_count = chunk_count
                st.session_state.sections = retrieval_agent.get_sections()

                st.success(f"✅ PDF processed! {chunk_count} chunks indexed.")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

    # Processing status
    if st.session_state.pdf_processed:
        st.success(f"📊 {st.session_state.chunk_count} chunks in database")

        # Mode toggle
        st.markdown("---")
        st.subheader("🎯 Mode")
        st.session_state.mode = st.radio(
            "Select mode:",
            ["Q&A Mode", "Quiz Mode"],
            index=0,
            help="Q&A Mode: Ask questions. Quiz Mode: Practice with questions.",
        )

        # Topic/section selector
        if st.session_state.sections:
            st.markdown("---")
            st.subheader("📑 Topics")
            selected_section = st.selectbox(
                "Filter by section:",
                ["All Topics"] + st.session_state.sections,
            )
    else:
        st.info("Upload a PDF to get started.")

    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_resource.clear()
        st.rerun()


# ── Helper functions ─────────────────────────────────────────────────────────


def _format_quiz_question(quiz_data: dict) -> str:
    """Format a quiz question dict into a readable message."""
    difficulty_icons = {"Basic": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    icon = difficulty_icons.get(quiz_data.get("difficulty", ""), "⚪")

    parts = [
        f"### 📝 Quiz Question {icon} {quiz_data.get('difficulty', '')}",
        "",
        f"**{quiz_data['question']}**",
        "",
    ]

    if quiz_data.get("options"):
        for option in quiz_data["options"]:
            parts.append(f"- {option}")
        parts.append("")

    parts.append(
        f"*Source: Page {quiz_data['source']['page']}, "
        f"Section: \"{quiz_data['source']['section']}\"*"
    )
    parts.append("")
    parts.append("Type your answer below! (e.g., 'My answer is A' or type your full answer)")

    return "\n".join(parts)


# ── Main chat area ───────────────────────────────────────────────────────────

st.header("💬 Chat")

if not st.session_state.pdf_processed:
    st.info(
        "👈 Upload a pharmacy course PDF in the sidebar to begin studying. "
        "I can help you understand concepts and practice with quizzes!"
    )
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show source panel for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📖 View Sources"):
                    for source in msg["sources"]:
                        text_preview = source["text"]
                        if len(text_preview) > 300:
                            text_preview = text_preview[:300] + "..."
                        st.markdown(
                            f"**Page {source['page_number']}** — "
                            f"*{source['section_title']}*\n\n"
                            f"> {text_preview}"
                        )

    # Display active quiz question as a card
    if st.session_state.current_quiz:
        quiz = st.session_state.current_quiz
        with st.container():
            st.markdown("---")
            difficulty_colors = {
                "Basic": "🟢",
                "Intermediate": "🟡",
                "Advanced": "🔴",
            }
            difficulty_icon = difficulty_colors.get(quiz.get("difficulty", ""), "⚪")
            st.markdown(
                f"### 📝 Quiz Question {difficulty_icon} {quiz.get('difficulty', '')}"
            )
            st.markdown(f"**{quiz['question']}**")

            if quiz.get("options"):
                st.markdown("**Options:**")
                for option in quiz["options"]:
                    st.markdown(f"- {option}")

            st.markdown(
                "*Type your answer below (e.g., 'My answer is A' or type your full answer).*"
            )
            st.markdown("---")

    # Chat input
    if user_input := st.chat_input("Ask a question or request a quiz..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process the message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                orchestrator = get_orchestrator()
                retrieval_agent = get_retrieval_agent()
                tutor_agent = get_tutor_agent()
                quiz_agent = get_quiz_agent()

                # Route the message
                route = orchestrator.route(user_input)

                response_text = ""
                sources = []

                if route["agent"] == "greeting":
                    response_text = orchestrator.get_greeting_response()

                elif route["agent"] == "tutor":
                    # Retrieve relevant chunks
                    chunks = retrieval_agent.search(user_input)
                    sources = chunks

                    # Generate answer
                    response_text = tutor_agent.answer(
                        question=user_input,
                        context_chunks=chunks,
                        chat_history=st.session_state.messages[:-1],
                    )

                elif route["agent"] == "quiz" and route["mode"] == "generate":
                    # Quiz generation mode
                    topic = user_input
                    chunks = retrieval_agent.search(topic)
                    sources = chunks

                    quiz_data = quiz_agent.generate_question(
                        topic=topic, context_chunks=chunks
                    )

                    if quiz_data:
                        st.session_state.current_quiz = quiz_data
                        response_text = _format_quiz_question(quiz_data)
                    else:
                        response_text = (
                            "I couldn't generate a quiz question for that topic. "
                            "Try rephrasing or choosing a different topic from the material."
                        )

                elif route["agent"] == "quiz" and route["mode"] == "evaluate":
                    # Quiz answer evaluation mode
                    if st.session_state.current_quiz:
                        quiz = st.session_state.current_quiz
                        chunks = retrieval_agent.search(quiz["question"])
                        sources = chunks

                        response_text = quiz_agent.evaluate_answer(
                            question=quiz["question"],
                            correct_answer=quiz["correct_answer"],
                            student_answer=user_input,
                            context_chunks=chunks,
                        )
                        # Clear the active quiz
                        st.session_state.current_quiz = None
                    else:
                        response_text = (
                            "There's no active quiz question to evaluate. "
                            "Ask me to quiz you on a topic first!"
                        )

                st.markdown(response_text)

                # Show sources
                if sources:
                    with st.expander("📖 View Sources"):
                        for source in sources:
                            text_preview = source["text"]
                            if len(text_preview) > 300:
                                text_preview = text_preview[:300] + "..."
                            st.markdown(
                                f"**Page {source['page_number']}** — "
                                f"*{source['section_title']}* "
                                f"(relevance: {source.get('relevance_score', 'N/A')})\n\n"
                                f"> {text_preview}"
                            )

        # Store assistant message
        msg_data = {"role": "assistant", "content": response_text}
        if sources:
            msg_data["sources"] = sources
        st.session_state.messages.append(msg_data)

        st.rerun()
