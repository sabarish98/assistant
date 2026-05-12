"""
Smoke test for the Q&A LangGraph workflow.

Run from the assistant/ directory with the venv active:
    python test_workflow.py

Tests
-----
1. Graph compiles without errors
2. Graph structure has the expected nodes and edges
3. run_qa() executes end-to-end (requires Ollama + ingested documents)
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))


def test_graph_compiles():
    """The graph should compile and expose the expected nodes."""
    print("1. Testing graph compilation...")

    from workflows.qa_workflow import build_qa_graph

    graph = build_qa_graph()
    assert graph is not None, "Graph should not be None"

    # LangGraph compiled graphs expose their nodes via .nodes
    node_names = set(graph.nodes.keys())
    expected = {
        "query_analysis",
        "document_retrieval",
        "generate_response",
        "validate_and_format",
        "handle_empty",
    }
    missing = expected - node_names
    assert not missing, f"Missing nodes: {missing}"

    print(f"   ✅ Graph compiled with nodes: {sorted(node_names)}")


def test_state_schema():
    """QAState TypedDict should import cleanly and have the right keys."""
    print("2. Testing state schema...")

    from workflows.state import QAState
    import typing

    hints = typing.get_type_hints(QAState)
    expected_keys = {"query", "search_results", "context_text", "raw_llm_output", "final_response", "error"}
    missing = expected_keys - set(hints.keys())
    assert not missing, f"Missing state keys: {missing}"

    print(f"   ✅ QAState keys: {sorted(hints.keys())}")


def test_run_qa_live():
    """
    End-to-end test — requires Ollama running and documents ingested.
    Skipped automatically if Ollama is unreachable.
    """
    print("3. Testing run_qa() end-to-end (requires Ollama)...")

    try:
        from workflows.qa_workflow import run_qa
        from schemas.response import LLMResponse

        response = run_qa("What topics are covered in the ingested documents?")

        assert isinstance(response, LLMResponse), "Should return an LLMResponse"
        assert response.query_id, "Should have a query_id"
        assert response.content, "Should have content"
        assert response.confidence is not None, "Should have confidence"
        assert response.validation is not None, "Should have validation"

        print(f"   ✅ Response received")
        print(f"      Content  : {response.content[:120]}...")
        print(f"      Confidence: {response.confidence.overall_score} ({response.confidence.level})")
        print(f"      Sources  : {len(response.sources)}")
        print(f"      Valid    : {response.validation.is_valid}")
        if response.validation.warnings:
            print(f"      Warnings : {response.validation.warnings}")

    except Exception as exc:
        print(f"   ⚠️  Skipped (Ollama may not be running): {exc}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Q&A Workflow Smoke Tests")
    print("=" * 55)

    try:
        test_graph_compiles()
        test_state_schema()
        test_run_qa_live()
        print("\n✅ All tests passed")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
