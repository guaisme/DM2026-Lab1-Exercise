from typing import List, Optional, Dict, Any
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        stop_words: Optional[str] = None,
        ngram_range: List[int] = [1, 1]
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from the loaded dataset.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        By default, stop words are NOT removed; pass `stop_words="english"` to filter common English words.
        The resulting document-term matrix is stored in `session.feature_matrix` and `session.feature_names`,
        which are required by subsequent filtering, dimensionality reduction, and similarity tools.
        Returns a summary dictionary containing sparsity metrics (`non_zero`, `total_elements`, `sparsity_pct`),
        vocabulary size, and a preview of feature names.

        Args:
            max_features: Optional maximum number of top vocabulary terms to keep based on term frequency.
            stop_words: Optional stop word removal setting (e.g., 'english' to drop common stop words; default None keeps all words).
            ngram_range: A 2-element list `[min_n, max_n]` specifying the lower and upper boundary of n-gram sizes (e.g., [1, 1] for unigrams, [1, 2] for unigrams and bigrams).

        Returns:
            A summary dictionary containing result_id, vocabulary size, feature preview, and sparsity metrics, or an error dict if input is invalid.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        if not isinstance(ngram_range, (list, tuple)) or len(ngram_range) != 2:
            return {"error": "ngram_range must be a list or tuple of exactly two integers, e.g., [1, 1]."}

        if not (isinstance(ngram_range[0], int) and isinstance(ngram_range[1], int)):
            return {"error": "ngram_range elements must both be integers."}

        if ngram_range[0] > ngram_range[1]:
            return {"error": f"Invalid ngram_range {ngram_range}: lower bound ({ngram_range[0]}) cannot be greater than upper bound ({ngram_range[1]})."}

        text_series = session.dataframe["text"].fillna("")

        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=tuple(ngram_range)
        )

        dtm = vectorizer.fit_transform(text_series)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "feature_names_preview": feature_names[:10],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "stop_words": stop_words, "ngram_range": ngram_range}, summary)

        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Calculates total word frequencies across all documents in the document-term matrix.

        `build_dtm_tool` must be run first to populate `session.feature_matrix` and `session.feature_names`.
        Sums term counts across all rows in the global document-term matrix.
        Returns a summary dictionary with the total vocabulary size and top N most frequent terms.
        Stores the complete sorted frequency report in `full_report` for `visualize_result_tool` to chart.

        Args:
            top_n: Number of top frequent terms to return in the summary dictionary (default 20).

        Returns:
            A summary dictionary containing result_id, vocabulary_size, and top_frequent_terms, or an error dict if DTM is missing.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        # Sum column counts across all rows in the matrix
        counts = session.feature_matrix.sum(axis=0).A1.tolist()

        df_freq = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        top_terms_dict = dict(zip(df_freq["term"].head(top_n), df_freq["frequency"].head(top_n)))

        result_id = session.next_result_id("term_freq")

        summary = {
            "result_id": result_id,
            "status": "success",
            "vocabulary_size": len(session.feature_names),
            "top_frequent_terms": top_terms_dict
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=df_freq
        )

        return summary

    return [build_dtm_tool, term_frequency_tool]
