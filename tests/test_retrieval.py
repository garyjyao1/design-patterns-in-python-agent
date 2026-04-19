from langchain_core.documents import Document

from retrieval_graph.retrieval import SimpleKeywordRetriever


def test_simple_keyword_retriever_returns_most_relevant_docs():
    retriever = SimpleKeywordRetriever(
        [
            Document(page_content="Singleton pattern ensures a class has one instance."),
            Document(page_content="Factory method creates objects without specifying class."),
            Document(page_content="Observer pattern defines one-to-many dependencies."),
        ]
    )

    docs = retriever.invoke("How does singleton ensure one instance?", k=2)

    assert len(docs) == 2
    assert "Singleton" in docs[0].page_content
    assert "Observer" in docs[1].page_content
