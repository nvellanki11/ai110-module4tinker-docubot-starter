"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Splitting documents into overlapping sliding-window chunks
- Building a vector retrieval index via Gemini embeddings (Phase 1)
- Retrieving relevant chunks by cosine similarity (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re
import math

class DocuBot:
    # Sliding-window chunking parameters (characters). The window needs to
    # be large enough to give the LLM real context but small enough that a
    # single embedding stays focused on one topic, so retrieval doesn't
    # dilute precision by averaging over an entire document. The overlap
    # keeps ideas that straddle a chunk boundary from getting split across
    # two embeddings and losing their surrounding context in both.
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

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

    def _chunk_text(self, text, chunk_size=None, overlap=None):
        """
        Split text into overlapping, fixed-size character windows.

        Each chunk starts `chunk_size - overlap` characters after the
        previous one, so consecutive chunks share `overlap` characters of
        context instead of cutting cleanly at the boundary. A short
        document that fits in a single window yields exactly one chunk.
        """
        chunk_size = chunk_size or self.CHUNK_SIZE
        overlap = overlap or self.CHUNK_OVERLAP
        stride = chunk_size - overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            if start + chunk_size >= len(text):
                break
            start += stride
        return chunks

    def build_index(self, documents):
        """
        Build a vector index: one Gemini embedding per chunk, where each
        document is split into overlapping sliding-window chunks. Chunking
        balances context (windows large enough to answer from) against
        precision (windows small enough that an embedding isn't averaged
        across unrelated sections of the document).

        Returns a list of (filename, chunk_text, embedding) tuples.
        Embeddings require an llm_client, so without one the index is
        empty and retrieve() has nothing to compare against.

        A single chunk that fails to embed (rate limit, transient network
        error) is skipped with a warning rather than aborting construction
        and losing every other chunk's embedding.
        """
        if self.llm_client is None:
            return []

        index = []
        for filename, text in documents:
            for chunk in self._chunk_text(text):
                try:
                    embedding = self.llm_client.embed_text(chunk)
                except RuntimeError as e:
                    print(f"Warning: skipping a chunk of {filename} in retrieval index ({e})")
                    continue
                index.append((filename, chunk, embedding))
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query_embedding, doc_embedding):
        """
        Cosine similarity between a query embedding and a chunk
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
        Embed the query, score every chunk by cosine similarity against
        its embedding, and return the top_k (score, filename, chunk_text)
        tuples sorted by similarity descending. Multiple chunks from the
        same file can appear if they're independently relevant.
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
        Score every chunk against the query and return the top_k
        (filename, chunk_text) pairs sorted by similarity descending.
        """
        scored = self.retrieve_with_scores(query, top_k=top_k)
        return [(filename, text) for _, filename, text in scored]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def _excerpt(self, query, text, max_chars=220):
        """
        Return a short, single-paragraph excerpt of a chunk's text,
        centered on the first query word found, instead of the full
        chunk.
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

        Guardrail: a chunk only counts as a match if its cosine
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
