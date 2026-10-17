import os
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

# Quotes from TEST_FIXTURE.md:
# Known Answer #17:
# "filtering_method="variance" and filtering_method="term_frequency" both keep zero terms for both categories on this fixture."
# "filtering_method="tfidf" is the one that leaves anything to mine: catA keeps only alpha (mean TF-IDF 0.6376 vs beta's 0.3676...); catB keeps only gamma (0.8143 vs delta's 0.5444)."
# "catA -> {alpha} support 3 (min_sup 1-3, 0 patterns at min_sup=4); catB -> {gamma} support 4 (min_sup 1-4)"
#
# Known Answer #20:
# "algorithm="topk" (PAMI.frequentPattern.topk.FAE): at k=1, k=2, and k=3 alike, both categories return exactly one pattern, catA -> {alpha} support 3, catB -> {gamma} support 4..."
# "algorithm="maxfpgrowth" (PAMI.frequentPattern.maximal.MaxFPGrowth): returns zero patterns for both categories, at both min_sup=1 and min_sup=4..."


def _setup_session_and_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert "error" not in load_res
    return session, mine_patterns_tool


def test_mine_patterns_known_answer_17_zero_terms():
    session, mine_patterns_tool = _setup_session_and_fixture()
    for cat in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["status"] == "success"
            assert res["n_kept_terms"] == 0
            assert res["n_patterns"] == 0
            assert res["patterns"] == []
            assert "note" in res
            assert session.artifacts[f"patterns_{cat}_{method}_fpgrowth"] == []


def test_mine_patterns_known_answer_17_tfidf_fpgrowth():
    session, mine_patterns_tool = _setup_session_and_fixture()

    # catA
    for ms in [1, 2, 3]:
        res = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": ms
        })
        assert res["status"] == "success"
        assert res["n_kept_terms"] == 1
        assert res["patterns"] == [{"pattern": ["alpha"], "support": 3}]
        assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == [{"pattern": ["alpha"], "support": 3}]

    res_catA_ms4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_ms4["status"] == "success"
    assert res_catA_ms4["patterns"] == []
    assert "note" in res_catA_ms4

    # catB
    for ms in [1, 2, 3, 4]:
        res = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": ms
        })
        assert res["status"] == "success"
        assert res["n_kept_terms"] == 1
        assert res["patterns"] == [{"pattern": ["gamma"], "support": 4}]
        assert session.artifacts["patterns_catB_tfidf_fpgrowth"] == [{"pattern": ["gamma"], "support": 4}]


def test_mine_patterns_known_answer_20_topk_and_maxfpgrowth():
    session, mine_patterns_tool = _setup_session_and_fixture()

    # topk
    for k_val in [1, 2, 3]:
        res_a = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_a["status"] == "success"
        assert res_a["patterns"] == [{"pattern": ["alpha"], "support": 3}]

        res_b = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_b["status"] == "success"
        assert res_b["patterns"] == [{"pattern": ["gamma"], "support": 4}]

    # maxfpgrowth
    for ms in [1, 4]:
        for cat in ["catA", "catB"]:
            res_max = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": ms
            })
            assert res_max["status"] == "success"
            assert res_max["patterns"] == []
            assert "note" in res_max


def test_mine_patterns_errors_and_file_cleanliness():
    files_before = set(os.listdir("."))
    session, mine_patterns_tool = _setup_session_and_fixture()

    # Run at least one successful pattern mining call
    res_ok = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_ok["status"] == "success"

    # Error handling assertions
    err_cat = mine_patterns_tool.invoke({
        "category_name": "catZ",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in err_cat

    err_method = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "bogus",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in err_method

    err_fpgrowth_nosup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" in err_fpgrowth_nosup

    err_topk_nok = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in err_topk_nok

    files_after = set(os.listdir("."))
    assert files_before == files_after
