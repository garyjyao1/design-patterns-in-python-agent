"""Main retrieval QA agent graph."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from retrieval_graph.retrieval import get_default_retriever
from retrieval_graph.state import State


@lru_cache(maxsize=1)
def _get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retrieve(state: State) -> State:
    """Retrieve relevant chunks for the incoming question."""
    question = state.get("question", "")
    docs = get_default_retriever().invoke(question, k=4)
    return {
        "question": question,
        "sources": [doc.metadata.get("source", "") for doc in docs],
        "retrieved_docs": [doc.page_content for doc in docs],
    }


def respond(state: State) -> State:
    """Generate answer from retrieved context. Falls back without API key."""
    question = state.get("question", "")
    doc_texts = state.get("retrieved_docs", [])
    context = "\n\n".join(doc_texts)

    if os.getenv("OPENAI_API_KEY"):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer questions about Python design patterns using only provided context. "
                    "If context is insufficient, clearly say so.",
                ),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}",
                ),
            ]
        )
        model = _get_chat_model()
        response = model.invoke(prompt.format_messages(question=question, context=context))
        if hasattr(response, "content"):
            answer = str(response.content)
        else:
            answer = str(response)
    else:
        snippets = "\n\n".join(doc_texts[:2])
        answer = (
            "OPENAI_API_KEY is not set, so this response is extractive. "
            "Most relevant snippets:\n\n"
            f"{snippets}"
        )

    return {
        "question": question,
        "answer": answer,
        "sources": state.get("sources", []),
    }


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("respond", respond)
builder.add_edge("__start__", "retrieve")
builder.add_edge("retrieve", "respond")
graph = builder.compile()
