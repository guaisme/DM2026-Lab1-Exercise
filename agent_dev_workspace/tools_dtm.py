from typing import List, Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap of a positional slice of the raw document-term matrix (DTM).

        `build_dtm_tool` must be run first to populate `session.feature_matrix` and `session.feature_names`.
        Slices the raw DTM up to `n_documents` rows and `n_terms` columns in existing order without sorting.
        Both `n_terms` and `n_documents` must be integers between 1 and 100 to keep the summary small on real datasets.
        Renders an annotated Seaborn heatmap and assigns the figure to `session.pending_figure`.

        Args:
            n_terms: Number of terms (columns) to slice from the DTM (default 20, max 100).
            n_documents: Number of documents (rows) to slice from the DTM (default 20, max 100).

        Returns:
            A dictionary containing result_id, matrix (nested list of Python ints), term_labels, and document_labels,
            or an error dict if DTM is missing or arguments are out of bounds (1 to 100).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        if not isinstance(n_terms, int) or not isinstance(n_documents, int):
            return {"error": "n_terms and n_documents must be integers."}

        if n_terms < 1 or n_terms > 100:
            return {"error": f"n_terms must be between 1 and 100 (got {n_terms})."}

        if n_documents < 1 or n_documents > 100:
            return {"error": f"n_documents must be between 1 and 100 (got {n_documents})."}

        sub_matrix = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        term_labels = list(session.feature_names[0:n_terms])
        doc_labels = [int(i) for i in range(sub_matrix.shape[0])]

        matrix_nested = [[int(val) for val in row] for row in sub_matrix]

        fig, ax = plt.subplots(figsize=(max(6, len(term_labels) * 0.5), max(4, len(doc_labels) * 0.4)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            xticklabels=term_labels,
            yticklabels=doc_labels,
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        ax.set_title(f"DTM Heatmap (First {len(doc_labels)} docs x {len(term_labels)} terms)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "matrix": matrix_nested,
            "term_labels": term_labels,
            "document_labels": doc_labels
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
