"""State definitions for the retrieval graph."""

from typing import TypedDict


class State(TypedDict, total=False):
    """Graph state."""

    question: str
    answer: str
    sources: list[str]
    retrieved_docs: list[str]
