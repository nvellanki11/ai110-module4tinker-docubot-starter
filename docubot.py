"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def _tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }
        """
        index = {}
        for filename, text in documents:
            for word in set(self._tokenize(text)):
                index.setdefault(word, []).append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query:
        count how many times each query word appears in the text.
        """
        query_words = self._tokenize(query)
        text_words = self._tokenize(text)

        score = 0
        for word in query_words:
            score += text_words.count(word)
        return score

    def retrieve(self, query, top_k=3):
        """
        Score every document against the query and return the top_k
        (filename, text) pairs sorted by score descending, skipping
        documents that scored 0 (no overlap with the query).
        """
        scored = []
        for filename, text in self.documents:
            score = self.score_document(query, text)
            if score > 0:
                scored.append((score, filename, text))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [(filename, text) for _, filename, text in scored]
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def _excerpt(self, query, text, max_chars=220):
        """
        Return a short, single-paragraph excerpt of text centered on the
        first query word found, instead of the full document.
        """
        lower_text = text.lower()
        match_pos = -1
        for word in self._tokenize(query):
            pos = lower_text.find(word)
            if pos != -1 and (match_pos == -1 or pos < match_pos):
                match_pos = pos

        start = max(0, match_pos - max_chars // 2) if match_pos != -1 else 0
        snippet = " ".join(text[start:start + max_chars].split())

        prefix = "..." if start > 0 else ""
        suffix = "..." if start + max_chars < len(text) else ""
        return f"{prefix}{snippet}{suffix}"

    def _confidence(self, query, text):
        """
        Fraction of distinct query words that appear anywhere in text.
        Used as a guardrail so a single incidental word match doesn't
        count as a real answer.
        """
        query_words = set(self._tokenize(query))
        if not query_words:
            return 0.0

        text_words = set(self._tokenize(text))
        matched = query_words & text_words
        return len(matched) / len(query_words)

    def answer_retrieval_only(self, query, top_k=3, min_confidence=0.5):
        """
        Phase 1 retrieval only mode.
        Returns a concise, numbered list of short excerpts (not full
        documents) with no LLM involved.

        Guardrail: a document only counts as a match if at least
        min_confidence of the query's distinct words actually appear in
        it. Otherwise we refuse rather than show a weak, likely
        irrelevant excerpt.
        """
        snippets = self.retrieve(query, top_k=top_k)
        confident_snippets = [
            (filename, text)
            for filename, text in snippets
            if self._confidence(query, text) >= min_confidence
        ]

        if not confident_snippets:
            return "I do not know based on these docs."

        formatted = []
        for i, (filename, text) in enumerate(confident_snippets, start=1):
            excerpt = self._excerpt(query, text)
            formatted.append(f"{i}. {filename}\n   {excerpt}")

        return "\n\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
