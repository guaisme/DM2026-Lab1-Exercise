import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    result = build_dtm_tool.invoke({})

    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5

    assert "count_vectorizer" in session.artifacts


def test_build_dtm_tool_no_dataset():
    session = SessionState()
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    result = build_dtm_tool.invoke({})

    assert "error" in result
    assert "No dataset loaded" in result["error"]


def test_term_frequency_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Load dataset & build DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Calculate term frequencies
    result = term_frequency_tool.invoke({"top_n": 20})

    # Assert known answer #14 values as mapping (unordered key comparison)
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert result["top_frequent_terms"] == expected_frequencies

    # Check stored full_report for alphabetical tie-breaking:
    # "always" (freq 7) and "gamma" (freq 7) tie; "always" comes before "gamma" alphabetically
    result_id = result["result_id"]
    df_report = session.results_store[result_id]["full_report"]
    top_two_terms = df_report["term"].head(2).tolist()
    assert top_two_terms == ["always", "gamma"]


def test_term_frequency_tool_no_dtm():
    session = SessionState()
    dtm_tools = make_tools(session)
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    result = term_frequency_tool.invoke({})

    assert "error" in result
    assert "No document-term matrix found" in result["error"]


def test_dtm_heatmap_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Test default arguments (known answer #21 full matrix 8x5)
    default_res = dtm_heatmap_tool.invoke({})
    assert default_res["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert default_res["document_labels"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert default_res["matrix"][0] == [3, 1, 1, 0, 0]
    assert default_res["matrix"][7] == [0, 0, 0, 0, 0]
    assert session.pending_figure is not None

    # Test n_terms=2, n_documents=3 slice
    sliced_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert sliced_res["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert sliced_res["term_labels"] == ["alpha", "always"]
    assert sliced_res["document_labels"] == [0, 1, 2]


def test_dtm_heatmap_tool_errors():
    session = SessionState()
    dtm_tools = make_tools(session)
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Error when no DTM loaded
    no_dtm_res = dtm_heatmap_tool.invoke({})
    assert "error" in no_dtm_res
    assert "No document-term matrix found" in no_dtm_res["error"]

    # Build DTM to test out-of-bounds error handling
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # n_terms = 0
    res_zero = dtm_heatmap_tool.invoke({"n_terms": 0})
    assert "error" in res_zero

    # n_terms = 101
    res_too_large = dtm_heatmap_tool.invoke({"n_terms": 101})
    assert "error" in res_too_large
