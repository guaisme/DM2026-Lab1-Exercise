import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags near-constant terms by variance for removal using population variance (ddof=0).

        Args:
            threshold: Variance cutoff threshold. Terms with variance >= threshold are kept.
                       Default is 0.0.

        Returns:
            Dict with result_id, filtering summary stats, and top kept/removed terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        if sp.issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
        else:
            variances = np.var(X, axis=0, ddof=0)

        # Ensure small floating point errors near 0 don't produce negative variances
        variances = np.maximum(variances, 0.0)

        df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        })

        df.sort_values(by=["variance", "term"], ascending=[False, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        kept_df = df[df["variance"] >= threshold]
        removed_df = df[df["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        session.artifacts["variance_kept_terms"] = kept_terms

        result_id = session.next_result_id("variance")

        top_kept = [
            {"term": row["term"], "variance": round(float(row["variance"]), 6)}
            for _, row in kept_df.head(10).iterrows()
        ]
        top_removed = [
            {"term": row["term"], "variance": round(float(row["variance"]), 6)}
            for _, row in removed_df.head(10).iterrows()
        ]

        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": float(threshold),
            "n_terms_total": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_df),
            "top_kept": top_kept,
            "top_removed": top_removed
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df)

        return summary

    def _validate_correlation_inputs(target_class: str):
        if session.feature_matrix is None or session.feature_names is None:
            return None, None, None, {"error": "No document-term matrix found. Run build_dtm_tool first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return None, None, None, {"error": "No dataframe or category_name column found in session."}

        X = session.feature_matrix
        df = session.dataframe
        feature_names = session.feature_names

        if len(df) != X.shape[0]:
            return None, None, None, {"error": "The document-term matrix is stale relative to dataframe. Run build_dtm_tool again."}

        valid_categories = sorted(df["category_name"].dropna().unique().tolist())
        if target_class not in valid_categories:
            return None, None, None, {"error": f"Target class '{target_class}' not found. Valid categories are: {valid_categories}"}

        y = (df["category_name"] == target_class).astype(float).values
        std_y = np.std(y, ddof=0)

        if std_y == 0.0:
            return None, None, None, {"error": f"Target class vector for '{target_class}' is constant (std dev is 0)."}

        return X, feature_names, y, None

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Pearson correlation filter between term counts and a target class (one-vs-rest).

        Args:
            target_class: The target category name to correlate with (1 vs rest).
            threshold: Correlation threshold. Keeps terms with |r| >= threshold. Default 0.0.

        Returns:
            Dict with result_id, filtering summary stats, and top kept/removed terms.
        """
        X, feature_names, y, err = _validate_correlation_inputs(target_class)
        if err is not None:
            return err

        mean_y = np.mean(y)
        std_y = np.std(y, ddof=0)

        if sp.issparse(X):
            mean_x = np.asarray(X.mean(axis=0)).ravel()
            mean_x2 = np.asarray(X.power(2).mean(axis=0)).ravel()
            std_x = np.sqrt(np.maximum(mean_x2 - (mean_x ** 2), 0.0))

            mean_xy = np.asarray(X.T.dot(y)).ravel() / X.shape[0]
            cov_xy = mean_xy - (mean_x * mean_y)
        else:
            mean_x = np.mean(X, axis=0)
            std_x = np.std(X, axis=0, ddof=0)
            cov_xy = np.mean(X * y[:, None], axis=0) - (mean_x * mean_y)

        denom = std_x * std_y
        r_scores = np.zeros(len(feature_names), dtype=float)
        valid_mask = denom > 1e-12
        r_scores[valid_mask] = cov_xy[valid_mask] / denom[valid_mask]
        r_scores = np.nan_to_num(r_scores, nan=0.0)

        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": r_scores,
            "abs_r": np.abs(r_scores)
        })

        report_df.sort_values(by=["abs_r", "term"], ascending=[False, True], inplace=True)
        report_df.reset_index(drop=True, inplace=True)

        kept_df = report_df[report_df["abs_r"] >= threshold]
        removed_df = report_df[report_df["abs_r"] < threshold]

        kept_terms = kept_df["term"].tolist()
        session.artifacts["pearson_kept_terms"] = kept_terms

        full_report = report_df[["term", "pearson_r"]].copy()

        result_id = session.next_result_id("pearson")

        top_kept = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 6)}
            for _, row in kept_df.head(10).iterrows()
        ]
        top_removed = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 6)}
            for _, row in removed_df.head(10).iterrows()
        ]

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": float(threshold),
            "n_terms_total": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_df),
            "top_kept": top_kept,
            "top_removed": top_removed
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=full_report)

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Spearman rank correlation filter between term counts and a target class (one-vs-rest).

        Args:
            target_class: The target category name to correlate with (1 vs rest).
            threshold: Correlation threshold. Keeps terms with |rho| >= threshold. Default 0.0.

        Returns:
            Dict with result_id, filtering summary stats, and top kept/removed terms.
        """
        X, feature_names, y, err = _validate_correlation_inputs(target_class)
        if err is not None:
            return err

        rank_y = rankdata(y, method="average")
        mean_ry = np.mean(rank_y)
        std_ry = np.std(rank_y, ddof=0)

        n_terms = len(feature_names)
        r_scores = np.zeros(n_terms, dtype=float)

        is_sparse = sp.issparse(X)
        if is_sparse:
            X_csc = X.tocsc()

        for j in range(n_terms):
            if is_sparse:
                x_col = X_csc[:, j].toarray().ravel()
            else:
                x_col = X[:, j].ravel()

            rank_x = rankdata(x_col, method="average")
            std_rx = np.std(rank_x, ddof=0)

            if std_rx > 1e-12 and std_ry > 1e-12:
                mean_rx = np.mean(rank_x)
                cov_rxy = np.mean(rank_x * rank_y) - (mean_rx * mean_ry)
                r_scores[j] = cov_rxy / (std_rx * std_ry)
            else:
                r_scores[j] = 0.0

        r_scores = np.nan_to_num(r_scores, nan=0.0)

        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": r_scores,
            "abs_r": np.abs(r_scores)
        })

        report_df.sort_values(by=["abs_r", "term"], ascending=[False, True], inplace=True)
        report_df.reset_index(drop=True, inplace=True)

        kept_df = report_df[report_df["abs_r"] >= threshold]
        removed_df = report_df[report_df["abs_r"] < threshold]

        kept_terms = kept_df["term"].tolist()
        session.artifacts["spearman_kept_terms"] = kept_terms

        full_report = report_df[["term", "spearman_r"]].copy()

        result_id = session.next_result_id("spearman")

        top_kept = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 6)}
            for _, row in kept_df.head(10).iterrows()
        ]
        top_removed = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 6)}
            for _, row in removed_df.head(10).iterrows()
        ]

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": float(threshold),
            "n_terms_total": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_df),
            "top_kept": top_kept,
            "top_removed": top_removed
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=full_report)

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
