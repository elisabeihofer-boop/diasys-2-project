# Final Project for Dialogue Systems 2 at the University of Gothenburg

In this project I built an idiom explainer application in SpeechState/XState that can explain idioms in context and give examples. I compared two versions of the system in terms of quality of the explanations: one using RAG and one using only an LLM.

## Repository Guide

- More detailed description of the project: `Diasys2_FinalProject_Report.pdf`

- Comparison dialogues between RAG and LLM-only system + scoring: `system_comparison.pdf` + `scores.pdf`

- Directories:
  - `dataset`: contains juypter notebook used to preprocess the IdiomX dataset
  - `rag/src/main.ts`: script to upload processed dataset to Qdrant in batches
  - `starter`: contains code for RAG dialogue system
  - `starter-llm`: contains code for LLM-only dialogue system

## Technologies
- TypeScript
- SpeechState / XState – Dialogue management
- Microsoft Azure – Speech recognition, speech synthesis, and intent recognition
- Ollama / Llama 3.2 – Idiom extraction and response generation
- Qdrant – Vector database for the RAG-based system
- Qwen3 Embedding – Semantic embeddings for idiom retrieval
