import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools
from tools_data import make_tools as make_data_tools

def setup_session_and_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    return session, variance_filter_tool

def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against TEST_FIXTURE #5 known answers."""
    session, variance_filter = setup_session_and_dtm()
    
    # Run variance_filter_tool with threshold=0.15
    res = variance_filter.invoke({"threshold": 0.15})
    
    assert res["status"] == "success"
    assert res["threshold"] == 0.15
    assert res["n_terms_total"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    
    # Check kept terms artifact
    assert session.artifacts.get("variance_kept_terms") == ["alpha", "gamma", "delta", "beta"]
    
    # Check full_report format and exact values
    full_report = session.results_store[res["result_id"]]["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "variance"]
    
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    expected_variances = [1.1875, 1.109375, 0.25, 0.234375, 0.109375]
    
    assert full_report["term"].tolist() == expected_terms
    assert full_report["variance"].tolist() == pytest.approx(expected_variances, abs=1e-6)
    
    # Check top_removed in summary
    assert len(res["top_removed"]) == 1
    assert res["top_removed"][0]["term"] == "always"
    assert res["top_removed"][0]["variance"] == pytest.approx(0.109375, abs=1e-6)

def test_variance_filter_tool_report_only():
    """Verify that variance_filter_tool does not mutate feature_matrix or feature_names."""
    session, variance_filter = setup_session_and_dtm()
    
    orig_shape = session.feature_matrix.shape
    orig_names = list(session.feature_names)
    
    variance_filter.invoke({"threshold": 0.5})
    
    assert session.feature_matrix.shape == orig_shape
    assert list(session.feature_names) == orig_names

def test_variance_filter_tool_dense_matrix():
    """Verify variance_filter_tool works identically on dense numpy ndarrays."""
    session, variance_filter = setup_session_and_dtm()
    
    # Convert sparse feature matrix to dense
    session.feature_matrix = session.feature_matrix.toarray()
    
    res = variance_filter.invoke({"threshold": 0.15})
    full_report = session.results_store[res["result_id"]]["full_report"]
    
    expected_variances = [1.1875, 1.109375, 0.25, 0.234375, 0.109375]
    assert full_report["variance"].tolist() == pytest.approx(expected_variances, abs=1e-6)

def test_variance_filter_tool_no_dtm_error():
    """Verify error dict when no DTM exists in session."""
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    variance_filter = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    res = variance_filter.invoke({"threshold": 0.0})
    assert "error" in res

def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against TEST_FIXTURE #6 known answers."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    build_dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")
    pearson_filter_tool = next(t for t in make_filtering_tools(session) if t.name == "pearson_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert res["status"] == "success"
    
    full_report = session.results_store[res["result_id"]]["full_report"]
    assert list(full_report.columns) == ["term", "pearson_r"]
    
    expected_terms = ["delta", "gamma", "beta", "alpha", "always"]
    expected_r = [1.0000, 0.8307, -0.7746, -0.6882, 0.3780]
    
    assert full_report["term"].tolist() == expected_terms
    assert full_report["pearson_r"].tolist() == pytest.approx(expected_r, abs=1e-4)

def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against TEST_FIXTURE #6 known answers."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    build_dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")
    spearman_filter_tool = next(t for t in make_filtering_tools(session) if t.name == "spearman_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert res["status"] == "success"
    
    full_report = session.results_store[res["result_id"]]["full_report"]
    assert list(full_report.columns) == ["term", "spearman_r"]
    
    expected_terms = ["delta", "gamma", "beta", "alpha", "always"]
    expected_rho = [1.0000, 0.9363, -0.7746, -0.7500, 0.3780]
    
    assert full_report["term"].tolist() == expected_terms
    assert full_report["spearman_r"].tolist() == pytest.approx(expected_rho, abs=1e-4)

def test_pearson_threshold_filtering():
    """Test pearson_filter_tool with threshold=0.7 (|r| >= 0.7)."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    build_dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")
    pearson_filter_tool = next(t for t in make_filtering_tools(session) if t.name == "pearson_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.7})
    assert res["n_kept"] == 3
    assert res["n_removed"] == 2
    assert session.artifacts.get("pearson_kept_terms") == ["delta", "gamma", "beta"]

def test_correlation_filter_errors():
    """Test error handling for unknown target_class, missing DTM, and stale DTM."""
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    pearson_filter = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    # 1. No DTM error
    res_no_dtm = pearson_filter.invoke({"target_class": "catB"})
    assert "error" in res_no_dtm
    
    # Load dataset and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    build_dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # 2. Unknown target_class error
    res_unknown = pearson_filter.invoke({"target_class": "catZ"})
    assert "error" in res_unknown
    assert "catZ" in res_unknown["error"]
    
    # 3. Stale DTM error (drop duplicates shrinks df from 8 to 6 rows, leaving 8-row DTM stale)
    check_duplicates_tool = next(t for t in make_data_tools(session) if t.name == "check_duplicates_tool")
    check_duplicates_tool.invoke({"drop": True})
    
    res_stale = pearson_filter.invoke({"target_class": "catB"})
    assert "error" in res_stale
