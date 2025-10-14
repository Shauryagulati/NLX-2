# Retrieval-Augmented Generation (RAG) System  
### From Naive Baseline to Advanced Query Rewriting & Reranking Pipelines

**Author:** Shaurya Gulati  
**Course:** Natural Language Processing (NLX Assignment 2)  
**Institution:** Carnegie Mellon University  
**Technologies:** Python, HuggingFace Transformers, Sentence-Transformers, Milvus Lite, Flan-T5, Cross-Encoder  

---

## Overview

This project implements, experiments with, and evaluates **Retrieval-Augmented Generation (RAG)** — a framework that combines **information retrieval** with **large language models (LLMs)** for producing **factually grounded, context-aware answers**.

The work progresses in **two phases**:

1. **Naive RAG Pipeline** — foundational baseline implementation.  
2. **Advanced RAG Pipeline** — incorporates **Query Rewriting** and **Cross-Encoder Reranking**.

The accompanying codebase and documentation demonstrate the architectural design, data preprocessing, model experimentation, and evaluation results — from concept to a production-ready RAG system.

---

## System Architecture

### Core Workflow
```
Documents → Preprocess & Chunk → Embedding Generation → Vector DB
↑ |
| ↓
Query → Query Encoding → Retriever → Generator → Final Answer
```


### Components

| Component | Description |
|------------|-------------|
| **Embedding Layer** | Uses `sentence-transformers` models: `all-MiniLM-L6-v2` (384d) and `all-mpnet-base-v2` (768d). |
| **Vector Database** | `Milvus Lite` – lightweight, production-oriented ANN vector store. |
| **Retriever** | Retrieves top-*k* semantically similar chunks using cosine similarity. |
| **Generator** | Uses `google/flan-t5-base` for context-grounded generation. |
| **Prompt Strategies** | Instruction, Persona, and Chain-of-Thought (CoT). |

---

## Dataset and Preprocessing

- Dataset: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)  
- Loaded from the `passages.parquet` partition (~3200 passages).  
- No missing values; passages range from **1–2515 characters** (avg: 390).  
- Chunked using **RecursiveCharacterTextSplitter** with `chunk_size=512`, `overlap=50`.  
- Stored **4430 vectorized chunks** in Milvus Lite.

---

## Phase 1: Naive RAG Implementation

### Steps
1. Data ingestion and chunking.  
2. Embedding generation using 384d and 768d Sentence Transformers.  
3. Vector storage in Milvus Lite.  
4. Retrieval of top-*k* chunks (tested with k=1,3,5).  
5. Generation using `flan-t5-base`.  

### Prompt Strategies Tested
- **Instruction:** Direct query-answer prompting.  
- **Persona:** “You are an expert encyclopedia...”  
- **Chain-of-Thought:** “Let’s think step-by-step.”  

### Results Summary

| Embedding | Prompt | Top-k | Exact Match | F1 Score |
|------------|--------|-------|--------------|-----------|
| 384d | Instruction | 5 | 68% | 77.0% |
| 768d | Persona | 3 | 72% | 81.0% |
| 768d | Instruction | 5 | 72% | 80.7% |
| Any | CoT | - | 0% | <15% |

**Key Insight:** The 768d `all-mpnet-base-v2` model + “Persona” prompt (k=3) delivered the best balance of accuracy and efficiency.

---

## Phase 2: Advanced RAG Implementation

### Query Rewriting
- Implemented using **Flan-T5** to generate **2 paraphrased queries**.  
- Expands the retrieval space to include alternative phrasings.  
- Improves recall but increases computational cost.

### Cross-Encoder Reranking
- Used `cross-encoder/ms-marco-MiniLM-L-6-v2`.  
- Reranks 10 retrieved passages based on semantic relevance.  
- Selects top 3 for final context generation.

### Evaluation Metrics
- **Exact Match (EM)**  
- **F1 Score**  
- **RAGAs Metrics:**  
  - Faithfulness  
  - Answer Relevancy  
  - Context Precision  
  - Context Recall

### Results

| Model | Technique | EM | F1 | Observation |
|--------|------------|----|----|--------------|
| Baseline | 768d Persona | 64% | 73.6% | Strong baseline |
| Advanced (Rewriting + Rerank) |  | 60% | 69.6% | Slightly lower — needs tuning |

**Observation:** Increased complexity didn’t yield better results without fine-tuning, showing that *“more advanced ≠ better by default.”*

---

## Evaluation Framework

**RAGAs** (Retrieval-Augmented Generation Automated Scoring) - automated evaluation using `gpt-4o-mini` as an impartial judge.  
Evaluates four orthogonal metrics:

| Metric | Description |
|---------|-------------|
| **Faithfulness** | Answer grounded in retrieved context |
| **Answer Relevancy** | Semantic alignment with question |
| **Context Precision** | Relevant contexts appear early |
| **Context Recall** | All necessary info retrieved |

---

## Insights & Takeaways

- **Bigger embeddings (768d)** significantly improve retrieval quality.  
- **Prompt engineering** is crucial - “Persona” prompts outperform others.  
- **Query rewriting and reranking** add latency and should be tuned before production.  
- Smaller models like **Flan-T5** here struggle with CoT prompting. 

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language Models** | Flan-T5, Cross-Encoder (MiniLM) |
| **Embeddings** | all-MiniLM-L6-v2, all-mpnet-base-v2 |
| **Vector DB** | Milvus Lite |
| **Evaluation** | HuggingFace SQuAD, RAGAs |
| **Frameworks** | LangChain, HuggingFace, Sentence-Transformers |
| **Visualization** | Matplotlib, Pandas |

---
