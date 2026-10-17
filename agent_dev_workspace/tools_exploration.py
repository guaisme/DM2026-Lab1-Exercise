import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.tools import tool


def make_tools(session) -> List[Any]:
    """Factory creating tools_exploration tools bound to the given session."""

    @tool
    def cosine_similarity_tool(doc1_idx: int, doc2_idx: int) -> Dict[str, Any]:
        """Compute cosine similarity between two documents in the loaded dataset.

        Uses the count vectorizer fitted by build_dtm_tool to transform the
        raw texts of doc1_idx and doc2_idx, then calculates cosine similarity.

        Args:
            doc1_idx: 0-based integer index of the first document.
            doc2_idx: 0-based integer index of the second document.

        Returns:
            Dict containing result_id, doc1_idx, doc2_idx, document previews,
            and similarity score.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        vectorizer = session.artifacts.get("count_vectorizer")
        if vectorizer is None:
            return {
                "error": "Count vectorizer not found in session.artifacts. Run build_dtm_tool first."
            }

        n_docs = len(session.dataframe)
        if not (0 <= doc1_idx < n_docs):
            return {
                "error": f"doc1_idx {doc1_idx} out of range (0 to {n_docs - 1})."
            }

        if not (0 <= doc2_idx < n_docs):
            return {
                "error": f"doc2_idx {doc2_idx} out of range (0 to {n_docs - 1})."
            }

        # Fetch texts positionally using iloc and handle potential missing/non-string values
        text1_raw = session.dataframe["text"].iloc[doc1_idx]
        text2_raw = session.dataframe["text"].iloc[doc2_idx]

        text1 = str(text1_raw) if text1_raw is not None and not (isinstance(text1_raw, float) and np.isnan(text1_raw)) else ""
        text2 = str(text2_raw) if text2_raw is not None and not (isinstance(text2_raw, float) and np.isnan(text2_raw)) else ""

        # Vectorize using the stored count vectorizer
        vectors = vectorizer.transform([text1, text2])
        v1 = vectors[0].toarray().ravel()
        v2 = vectors[1].toarray().ravel()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            similarity = float(np.dot(v1, v2) / (norm1 * norm2))

        similarity = round(similarity, 6)

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "status": "success",
            "doc1_idx": doc1_idx,
            "doc2_idx": doc2_idx,
            "doc1_preview": text1[:80],
            "doc2_preview": text2[:80],
            "similarity": similarity,
        }

        session.store_result("cosine_similarity_tool", args={"doc1_idx": doc1_idx, "doc2_idx": doc2_idx}, summary=summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Compute term-vs-term Pearson correlation matrix over the top-N terms by variance.

        Generates a heatmap of feature correlations and sets session.pending_figure.

        Args:
            top_n: Number of top terms by variance to include (bounded 1 to 30, default 20).

        Returns:
            Dict containing result_id, term_names, correlation_matrix, and list of constant_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix not built. Run build_dtm_tool first."
            }

        if not (1 <= top_n <= 30):
            return {"error": "top_n must be between 1 and 30."}

        X = session.feature_matrix
        feature_names = list(session.feature_names)

        # Compute population variance directly without full matrix densification
        if hasattr(X, "toarray"):
            # Sparse matrix: mean of squares minus square of mean
            mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            sq_mean = np.asarray(X.mean(axis=0)).ravel() ** 2
            variances = mean_sq - sq_mean
        else:
            # Dense numpy array
            variances = np.var(X, axis=0, ddof=0)

        # Stable sort: variance descending, term name ascending
        # Tuple sort key: (-variance, term_name)
        sorted_indices = sorted(
            range(len(feature_names)),
            key=lambda i: (-variances[i], feature_names[i])
        )

        selected_indices = sorted_indices[:top_n]
        selected_terms = [feature_names[i] for i in selected_indices]
        selected_vars = variances[selected_indices]

        # Densify ONLY selected columns
        X_slice = X[:, selected_indices]
        dense_slice = X_slice.toarray() if hasattr(X_slice, "toarray") else np.asarray(X_slice)

        # Compute Pearson correlation matrix
        corr = np.corrcoef(dense_slice, rowvar=False)

        # Handle scalar (1x1) output from np.corrcoef
        if np.ndim(corr) == 0:
            corr = np.array([[corr]])

        constant_terms = []
        n_selected = len(selected_terms)

        # Detect constant terms and clean NaNs
        for idx in range(n_selected):
            if selected_vars[idx] == 0.0 or np.isnan(corr[idx, idx]):
                constant_terms.append(selected_terms[idx])

        corr_cleaned = np.nan_to_num(corr, nan=0.0)
        for idx in range(n_selected):
            corr_cleaned[idx, idx] = 1.0

        corr_rounded = [
            [round(float(val), 4) for val in row]
            for row in corr_cleaned
        ]

        # Build Seaborn heatmap
        fig_size = max(6, min(12, n_selected * 0.5 + 3))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        sns.heatmap(
            corr_cleaned,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title(f"Top-{n_selected} Term Correlation Matrix")
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "status": "success",
            "top_n": n_selected,
            "term_names": selected_terms,
            "correlation_matrix": corr_rounded,
            "constant_terms": constant_terms,
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            args={"top_n": top_n},
            summary=summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
