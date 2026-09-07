"""
Streamlit frontend for the DocuBot tinker activity.

Lets you interactively test the same three modes as main.py:
1. Naive LLM generation over all docs (Phase 0)
2. Retrieval only (Phase 1)
3. RAG: retrieval plus LLM generation (Phase 2)
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from docubot import DocuBot
from llm_client import GeminiClient
from dataset import SAMPLE_QUERIES


@st.cache_resource(show_spinner="Loading docs and building index...")
def load_bot():
    """
    Builds the GeminiClient and DocuBot once per server process. Cached
    because embedding the whole corpus/ folder on every rerun would be slow
    and burn API quota.
    """
    try:
        client = GeminiClient()
        has_llm = True
    except RuntimeError as exc:
        client = None
        has_llm = False
        st.session_state["llm_error"] = str(exc)

    bot = DocuBot(llm_client=client)
    return bot, has_llm


st.set_page_config(page_title="DocuBot", page_icon="📚")
st.title("📚 DocuBot")

bot, has_llm = load_bot()

if not has_llm:
    st.warning(
        "LLM features are disabled: "
        + st.session_state.get("llm_error", "no GEMINI_API_KEY")
        + "\n\nRetrieval relies on Gemini embeddings too, so it is unavailable as well."
    )

mode_options = {
    "Naive LLM over full docs (no retrieval)": "naive",
    "Retrieval only (vector search, no generation)": "retrieval",
    "RAG (retrieval + LLM)": "rag",
}
mode_label = st.radio("Mode", list(mode_options.keys()), disabled=not has_llm)
mode = mode_options[mode_label]

with st.sidebar:
    st.header("Query")
    sample = st.selectbox("Sample queries", [""] + SAMPLE_QUERIES)
    query = st.text_area("Or type your own query", value=sample)
    top_k = st.slider("top_k (chunks to retrieve)", min_value=1, max_value=10, value=3)
    run = st.button("Run", type="primary", disabled=not has_llm)

if run:
    if not query.strip():
        st.error("Enter a query first.")
    else:
        with st.spinner("Thinking..."):
            if mode == "naive":
                all_text = bot.full_corpus_text()
                answer = bot.llm_client.naive_answer_over_full_docs(query, all_text)
                st.subheader("Answer")
                st.write(answer)

            elif mode == "retrieval":
                answer = bot.answer_retrieval_only(query, top_k=top_k)
                st.subheader("Retrieved snippets")
                st.text(answer)

            elif mode == "rag":
                answer = bot.answer_rag(query, top_k=top_k)
                st.subheader("Answer")
                st.write(answer)

                with st.expander("Retrieved snippets used"):
                    for filename, text in bot.retrieve(query, top_k=top_k):
                        st.markdown(f"**{filename}**")
                        st.text(text)
