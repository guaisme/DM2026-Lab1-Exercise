import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools
from tools_dtm import make_tools as make_dtm_tools


def test_list_files_tool_newdataset():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["status"] == "success"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
    assert result["result_id"] in session.results_store


def test_list_files_tool_traversal_blocked():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "../.."})

    assert "error" in result
    assert "outside repository" in result["error"].lower()


def test_list_files_tool_missing_folder():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "this_folder_does_not_exist"})

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_check_missing_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Error when no dataset is loaded
    no_df_res = check_missing_tool.invoke({})
    assert "error" in no_df_res

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test check_missing_tool against TEST_FIXTURE #2
    result = check_missing_tool.invoke({})
    assert result["status"] == "success"
    assert result["missing_text_count"] == 1
    assert result["missing_text_indices_preview"] == [7]
    assert result["missing_counts"]["text"] == 1


def test_check_duplicates_tool():
    # Test error when no dataset loaded
    session_no_df = SessionState()
    data_tools_no_df = make_tools(session_no_df)
    check_duplicates_no_df = next(t for t in data_tools_no_df if t.name == "check_duplicates_tool")
    no_df_res = check_duplicates_no_df.invoke({})
    assert "error" in no_df_res

    # Test drop=False against TEST_FIXTURE #3
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    data_tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res_no_drop = check_duplicates_tool1.invoke({"drop": False})
    assert res_no_drop["status"] == "success"
    assert res_no_drop["n_duplicates"] == 1
    assert res_no_drop["n_rows_involved"] == 2
    assert res_no_drop["duplicate_indices_preview"] == [5, 6]
    assert res_no_drop["rows_before"] == 8
    assert res_no_drop["rows_after"] == 8
    assert len(session1.dataframe) == 8
    assert "note" not in res_no_drop

    # Test drop=True (keep=False drops every copy, leaving 6 rows)
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    data_tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res_drop = check_duplicates_tool2.invoke({"drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["rows_before"] == 8
    assert res_drop["rows_after"] == 6
    assert len(session2.dataframe) == 6
    assert "always gamma delta" not in session2.dataframe["text"].values
    assert "note" not in res_drop

    # Test staleness note when DTM exists and drop=True
    session3 = SessionState()
    premade_tools3, _ = load_workspace_tools("premade_tools", session3)
    load_dataset_tool3 = next(t for t in premade_tools3 if t.name == "load_dataset_tool")
    data_tools3 = make_tools(session3)
    dtm_tools3 = make_dtm_tools(session3)
    check_duplicates_tool3 = next(t for t in data_tools3 if t.name == "check_duplicates_tool")
    build_dtm_tool3 = next(t for t in dtm_tools3 if t.name == "build_dtm_tool")

    load_dataset_tool3.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool3.invoke({})

    res_drop_stale = check_duplicates_tool3.invoke({"drop": True})
    assert res_drop_stale["status"] == "success"
    assert "note" in res_drop_stale
    assert "stale" in res_drop_stale["note"].lower()


def test_sample_data_tool():
    # Test error when no dataset loaded
    session_no_df = SessionState()
    data_tools_no_df = make_tools(session_no_df)
    sample_data_no_df = next(t for t in data_tools_no_df if t.name == "sample_data_tool")
    assert "error" in sample_data_no_df.invoke({})

    # Load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Bounds validation (n=0 and n=9)
    assert "error" in sample_data_tool.invoke({"n": 0})
    assert "error" in sample_data_tool.invoke({"n": 9})

    # TEST_FIXTURE #13: n=4, random_state=42 -> sampled_indices == [1, 5, 0, 7]
    res_preview = sample_data_tool.invoke({"n": 4, "random_state": 42, "replace_working": False})
    assert res_preview["status"] == "success"
    assert res_preview["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 8
    assert "note" not in res_preview

    # replace_working=True shrinks session.dataframe to 4 rows
    res_replace = sample_data_tool.invoke({"n": 4, "random_state": 42, "replace_working": True})
    assert res_replace["status"] == "success"
    assert res_replace["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4
    assert "note" not in res_replace

    # replace_working=True with existing DTM -> staleness note
    session_dtm = SessionState()
    premade_tools_dtm, _ = load_workspace_tools("premade_tools", session_dtm)
    load_dataset_tool_dtm = next(t for t in premade_tools_dtm if t.name == "load_dataset_tool")
    data_tools_dtm = make_tools(session_dtm)
    dtm_tools = make_dtm_tools(session_dtm)

    sample_data_tool_dtm = next(t for t in data_tools_dtm if t.name == "sample_data_tool")
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool_dtm.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    res_stale = sample_data_tool_dtm.invoke({"n": 4, "random_state": 42, "replace_working": True})
    assert "note" in res_stale
    assert "stale" in res_stale["note"].lower()


def test_tokenize_tool():
    # Test error when no dataset loaded
    session_no_df = SessionState()
    data_tools_no_df = make_tools(session_no_df)
    tokenize_no_df = next(t for t in data_tools_no_df if t.name == "tokenize_tool")
    assert "error" in tokenize_no_df.invoke({})

    # Load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    result = tokenize_tool.invoke({})
    assert result["status"] == "success"
    assert result["n_documents"] == 8

    # TEST_FIXTURE #10: row 0 unigrams and row 7 empty list
    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert unigrams[7] == []

    # Ensure full unigram lists are not in summary
    assert "unigrams" not in result

    # Custom test for mixed-case lowercasing ("Alpha BETA" -> ["alpha", "beta"])
    session_case = SessionState()
    session_case.dataframe = pd.DataFrame({
        "text": ["Alpha BETA"],
        "category": [0],
        "category_name": ["catA"]
    })
    data_tools_case = make_tools(session_case)
    tokenize_case = next(t for t in data_tools_case if t.name == "tokenize_tool")

    tokenize_case.invoke({})
    assert session_case.dataframe["unigrams"].iloc[0] == ["alpha", "beta"]
