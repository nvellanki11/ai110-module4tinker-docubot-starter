"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a vector retrieval index via Gemini embeddings (Phase 1)
- Retrieving relevant snippets by cosine similarity (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re
import math

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: Gemini client used both for LLM answers and for
            computing the embeddings that back vector retrieval. Without
            it, no index can be built and retrieve() returns nothing.
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a vector index (implemented in Phase 1)
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
        Build a vector index: one Gemini embedding per document.

        Returns a list of (filename, text, embedding) tuples. Embeddings
        require an llm_client, so without one the index is empty and
        retrieve() has nothing to compare against.

        A single document that fails to embed (rate limit, transient
        network error) is skipped with a warning rather than aborting
        construction and losing every other document's embedding.
        """
        if self.llm_client is None:
            return []

        index = []
        for filename, text in documents:
            try:
                embedding = self.llm_client.embed_text(text)
            except RuntimeError as e:
                print(f"Warning: skipping {filename} in retrieval index ({e})")
                continue
            index.append((filename, text, embedding))
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query_embedding, doc_embedding):
        """
        Cosine similarity between a query embedding and a document
        embedding. Ranges from -1 to 1; higher means more semantically
        similar.
        """
        dot = sum(x * y for x, y in zip(query_embedding, doc_embedding))
        query_norm = math.sqrt(sum(x * x for x in query_embedding))
        doc_norm = math.sqrt(sum(y * y for y in doc_embedding))

        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return dot / (query_norm * doc_norm)

    def retrieve_with_scores(self, query, top_k=3):
        """
        Embed the query, score every document by cosine similarity
        against its embedding, and return the top_k
        (score, filename, text) tuples sorted by similarity descending.
        """
        if not self.index:
            return []

        query_embedding = self.llm_client.embed_text(query)

        scored = [
            (self.score_document(query_embedding, embedding), filename, text)
            for filename, text, embedding in self.index
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def retrieve(self, query, top_k=3):
        """
        Score every document against the query and return the top_k
        (filename, text) pairs sorted by similarity descending.
        """
        scored = self.retrieve_with_scores(query, top_k=top_k)
        return [(filename, text) for _, filename, text in scored]

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

    def answer_retrieval_only(self, query, top_k=3, min_similarity=0.6):
        """
        Phase 1 retrieval only mode.
        Returns a concise, numbered list of short excerpts (not full
        documents) with no LLM involved.

        Guardrail: a document only counts as a match if its cosine
        similarity to the query embedding is at least min_similarity.
        Otherwise we refuse rather than show a weak, likely irrelevant
        excerpt.
        """
        scored = self.retrieve_with_scores(query, top_k=top_k)
        confident_snippets = [
            (filename, text)
            for score, filename, text in scored
            if score >= min_similarity
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
