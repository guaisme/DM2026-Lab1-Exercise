import tempfile
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/patterns for a specific category using PAMI.

        Filters category vocabulary by variance, term_frequency, or tfidf,
        converts surviving terms into a transactional database, and runs
        fpgrowth, topk (FAE), or maxfpgrowth.

        Args:
            category_name: Target category label to filter session.dataframe on.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: Pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of top patterns to find (required for 'topk').

        Returns:
            Dict containing mining summary, count of patterns, and pattern list.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session"}

        if "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame missing 'category_name' column"}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame missing 'text' column"}

        valid_categories = list(session.dataframe["category_name"].unique())
        if category_name not in valid_categories:
            return {
                "error": f"Unknown category '{category_name}'. Available categories: {valid_categories}"
            }

        valid_methods = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_methods:
            return {
                "error": f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_methods}"
            }

        valid_algorithms = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {
                "error": f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}"
            }

        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                return {
                    "error": f"algorithm '{algorithm}' requires a positive integer min_sup"
                }

        if algorithm == "topk":
            if k is None or k <= 0:
                return {"error": "algorithm 'topk' requires a positive integer k"}

        full_args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        def _empty_result(note: str) -> Dict[str, Any]:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_kept_terms": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": note,
            }
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            session.store_result("mine_patterns_tool", full_args, summary)
            return summary

        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        texts = cat_df["text"].fillna("").tolist()

        try:
            vectorizer = CountVectorizer(stop_words="english")
            dtm_sparse = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Raised if all documents are empty or consist entirely of stopwords
            return _empty_result("no terms survived stopword removal")

        if len(feature_names) == 0:
            return _empty_result("no terms survived filtering")

        dtm_df = pd.DataFrame(dtm_sparse.toarray(), columns=feature_names)

        # Apply filtering matching text_filtering_helpers.py exactly
        if filtering_method == "variance":
            metric_series = dtm_df.var(axis=0).sort_values(ascending=False)
            lower = np.percentile(metric_series, 5)
            upper = np.percentile(metric_series, 95)
            kept_terms = metric_series[(metric_series >= lower) & (metric_series <= upper)].index.tolist()

        elif filtering_method == "term_frequency":
            metric_series = dtm_df.sum(axis=0).sort_values(ascending=False)
            lower = np.percentile(metric_series, 5)
            upper = np.percentile(metric_series, 95)
            kept_terms = metric_series[(metric_series >= lower) & (metric_series <= upper)].index.tolist()

        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm_df.values)
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            metric_series = pd.Series(tfidf_scores, index=dtm_df.columns)
            lower = np.percentile(metric_series, 20)
            kept_terms = metric_series[metric_series >= lower].index.tolist()

        n_kept_terms = len(kept_terms)

        if n_kept_terms == 0:
            return _empty_result("no terms survived filtering")

        filtered_dtm_df = dtm_df[kept_terms]

        patterns_raw = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/trans_db.txt"
            
            # Convert DF to Transactional DB
            converter = DF2DB(filtered_dtm_df)
            converter.convert2TransactionalDatabase(db_path, ">=", 1)

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(iFile=db_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=db_path, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                patterns_df.columns = [str(c).lower() for c in patterns_df.columns]
                pat_col = "patterns" if "patterns" in patterns_df.columns else patterns_df.columns[0]
                sup_col = "support" if "support" in patterns_df.columns else patterns_df.columns[1]

                for _, row in patterns_df.iterrows():
                    pat_str = str(row[pat_col])
                    terms = pat_str.strip().split()
                    sup_val = int(row[sup_col])
                    patterns_raw.append({"pattern": terms, "support": sup_val})

        # Sort patterns: descending by support, then ascending by pattern string representation for tie-breaking
        patterns_raw.sort(key=lambda x: (-x["support"], " ".join(x["pattern"])))

        session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = patterns_raw

        res_id = session.next_result_id("patterns")
        n_patterns = len(patterns_raw)

        top_patterns = patterns_raw[:20]

        summary = {
            "result_id": res_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_kept_terms": n_kept_terms,
            "n_patterns": n_patterns,
            "patterns": top_patterns,
        }

        if n_patterns == 0:
            summary["note"] = "no patterns found"

        session.store_result("mine_patterns_tool", full_args, summary)

        return summary

    return [mine_patterns_tool]
