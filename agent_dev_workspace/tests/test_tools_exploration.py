import pytest
import math
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools


def _setup_session():
    """Helper to create session and populate with fixture dataset and fitted DTM."""
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load workspace tools for DTM construction
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = {t.name: t for t in make_dtm_tools(session)}

    # 1. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM with default parameters
    dtm_tools["build_dtm_tool"].invoke({})

    # 3. Create exploration tools
    exp_tools = {t.name: t for t in make_tools(session)}

    return session, exp_tools["cosine_similarity_tool"]


def _setup_corr_session():
    """Helper to create session and return correlation tool along with session."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = {t.name: t for t in make_dtm_tools(session)}

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools["build_dtm_tool"].invoke({})

    exp_tools = {t.name: t for t in make_tools(session)}

    return session, exp_tools["feature_correlation_matrix_tool"]


def test_cosine_similarity_known_answers():
    """Quote TEST_FIXTURE #16:
    | Doc pair | Cosine similarity |
    | 0 vs 1 (both catA, near-identical text) | 0.9847 |
    | 0 vs 3 (catA vs catB, share only 'always') | 0.0909 |
    | 3 vs 4 (both catB, near-identical text) | 0.9847 |
    | 0 vs 7 (catA vs the empty-text row) | 0.0 |
    """
    _, sim_tool = _setup_session()

    # (0, 1) = 0.9847
    res_0_1 = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # (0, 3) = 0.0909
    res_0_3 = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)

    # (3, 4) = 0.9847
    res_3_4 = sim_tool.invoke({"doc1_idx": 3, "doc2_idx": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # (0, 7) = 0.0 exactly and not NaN
    res_0_7 = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["similarity"] == 0.0
    assert not math.isnan(res_0_7["similarity"])


def test_cosine_similarity_symmetry_and_self():
    _, sim_tool = _setup_session()

    # Symmetry: (3, 0) == (0, 3)
    res_0_3 = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 3})
    res_3_0 = sim_tool.invoke({"doc1_idx": 3, "doc2_idx": 0})
    assert res_3_0["similarity"] == pytest.approx(res_0_3["similarity"], abs=1e-6)

    # Self-similarity: (0, 0) == 1.0
    res_0_0 = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 0})
    assert res_0_0["similarity"] == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_errors():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    exp_tools = {t.name: t for t in make_tools(session)}
    sim_tool = exp_tools["cosine_similarity_tool"]

    # 1. Error: count_vectorizer missing (dataset loaded, but DTM not built)
    res_no_vec = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 1})
    assert "error" in res_no_vec

    # Now build DTM to test index errors
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = {t.name: t for t in make_dtm_tools(session)}
    dtm_tools["build_dtm_tool"].invoke({})

    # 2. Error: doc index -1
    res_neg = sim_tool.invoke({"doc1_idx": -1, "doc2_idx": 0})
    assert "error" in res_neg

    # 3. Error: doc index 8 (out of bounds for 8 docs)
    res_oob = sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 8})
    assert "error" in res_oob


def test_feature_correlation_matrix_known_answers():
    """Quote TEST_FIXTURE #19:
    All 5 terms survive the top-20 cutoff (there are only 5 to begin with).
    By variance, descending: alpha, gamma, delta, beta, always.
    Full correlation matrix, rounded to 4 decimals:
    | | alpha | gamma | delta | beta | always |
    | alpha | 1.0000 | -0.5718 | -0.6882 | 0.8885 | 0.2601 |
    | gamma | -0.5718 | 1.0000 | 0.8307 | -0.6435 | 0.3140 |
    | delta | -0.6882 | 0.8307 | 1.0000 | -0.7746 | 0.3780 |
    | beta | 0.8885 | -0.6435 | -0.7746 | 1.0000 | 0.2928 |
    | always | 0.2601 | 0.3140 | 0.3780 | 0.2928 | 1.0000 |
    """
    session, corr_tool = _setup_corr_session()

    res = corr_tool.invoke({})
    assert res["status"] == "success"
    assert res["term_names"] == ["alpha", "gamma", "delta", "beta", "always"]
    assert res["constant_terms"] == []
    assert session.pending_figure is not None

    terms = res["term_names"]
    matrix = res["correlation_matrix"]

    # Check symmetry and diagonal = 1.0
    for i in range(len(terms)):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-4)
        for j in range(len(terms)):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-4)

    expected_matrix = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]

    for i in range(len(terms)):
        assert matrix[i] == pytest.approx(expected_matrix[i], abs=1e-4)


def test_feature_correlation_matrix_top_n_and_errors():
    session, corr_tool = _setup_corr_session()

    # top_n=2 -> term_names == ["alpha", "gamma"] and 2x2 matrix
    res_top2 = corr_tool.invoke({"top_n": 2})
    assert res_top2["status"] == "success"
    assert res_top2["term_names"] == ["alpha", "gamma"]
    assert len(res_top2["correlation_matrix"]) == 2
    assert len(res_top2["correlation_matrix"][0]) == 2

    # Error: top_n=0
    res_zero = corr_tool.invoke({"top_n": 0})
    assert "error" in res_zero

    # Error: top_n=31
    res_oob = corr_tool.invoke({"top_n": 31})
    assert "error" in res_oob

    # Error: no DTM
    fresh_session = SessionState()
    exp_tools = {t.name: t for t in make_tools(fresh_session)}
    res_no_dtm = exp_tools["feature_correlation_matrix_tool"].invoke({})
    assert "error" in res_no_dtm
