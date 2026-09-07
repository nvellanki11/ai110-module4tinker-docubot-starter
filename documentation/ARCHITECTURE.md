# DocuBot Architecture

DocuBot supports three modes, all starting from the same developer question and
converging on a printed answer. Modes 2 and 3 share a retrieval path (query
embedding → cosine similarity against a pre-built chunk index); mode 3 feeds the
retrieved snippets back into Gemini, while mode 2 stops at the raw excerpts.
Mode 1 skips retrieval entirely and hands the whole corpus to Gemini.

A rendered graphic of this diagram is checked into the repo at
[`images/architecture.png`](images/architecture.png):

![DocuBot architecture diagram](images/architecture.png)

<details>
<summary>Mermaid source</summary>

```mermaid
flowchart TD
    Query["Developer question"]
    Mode{"Selected mode"}

    Query --> Mode

    subgraph Mode1["1 · Naive LLM"]
        direction TB
        M1Corpus["Load full docs corpus\n(full_corpus_text)"]
        M1Gen["Gemini generate_content\n(naive_answer_over_full_docs)"]
        M1Corpus --> M1Gen
    end

    subgraph Mode2["2 · Retrieval only"]
        direction TB
        M2Embed["Embed query\n(embed_text)"]
        M2Score["Score chunks by cosine similarity\nvs. cached chunk embeddings"]
        M2Filter["Keep chunks >= min_similarity\n(answer_retrieval_only)"]
        M2Excerpt["Build short excerpts\n(_excerpt)"]
        M2Embed --> M2Score --> M2Filter --> M2Excerpt
    end

    subgraph Mode3["3 · RAG"]
        direction TB
        M3Embed["Embed query\n(embed_text)"]
        M3Score["Score chunks by cosine similarity"]
        M3Top["Take top_k snippets\n(retrieve)"]
        M3Gen["Gemini generate_content\nwith snippets as context\n(answer_from_snippets)"]
        M3Embed --> M3Score --> M3Top --> M3Gen
    end

    Mode -->|"1"| Mode1
    Mode -->|"2"| Mode2
    Mode -->|"3"| Mode3

    Index[("Chunk index\n(sliding-window chunks\n+ Gemini embeddings,\ndisk-cached)")]
    Index -.-> M2Score
    Index -.-> M3Score

    M1Gen --> Output["Answer printed to console"]
    M2Excerpt --> Output
    M3Gen --> Output
```

</details>
