from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """Lists files and folders in a specified repository subdirectory without recursing.

        Helps discover available dataset files and directory structures before loading data.
        Restricted to the repository root and subdirectories to prevent directory traversal.

        Args:
            subdirectory: Optional relative path from repository root (e.g., 'newdataset').

        Returns:
            A dictionary containing result_id, status, sorted list of directory names,
            and sorted list of file names, or an error dictionary if path is invalid.
        """
        base_dir = Path(__file__).resolve().parent.parent

        if subdirectory:
            target_dir = (base_dir / subdirectory).resolve()
        else:
            target_dir = base_dir

        try:
            target_dir.relative_to(base_dir)
        except ValueError:
            return {"error": "Access denied: requested path is outside repository base directory."}

        if not target_dir.exists() or not target_dir.is_dir():
            return {"error": f"Directory not found: '{subdirectory or '.'}'."}

        directories = []
        files = []

        for item in target_dir.iterdir():
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "status": "success",
            "directories": directories,
            "files": files
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values across the loaded working dataset.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        Identifies NaN/null values as well as empty or whitespace-only text strings.
        Returns missing value counts per column and a preview of row indices with missing text.

        Returns:
            A summary dictionary containing result_id, missing counts per column,
            and row indices of documents with empty text, or an error dict if no dataset is loaded.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe

        missing_counts = {}
        for col in df.columns:
            col_series = df[col]
            is_na = col_series.isna()
            if col_series.dtype == object or isinstance(col_series.dtype, pd.StringDtype):
                is_empty = col_series.fillna("").astype(str).str.strip() == ""
                missing_mask = is_na | is_empty
            else:
                missing_mask = is_na
            missing_counts[col] = int(missing_mask.sum())

        missing_text_indices = []
        if "text" in df.columns:
            text_series = df["text"]
            is_missing_text = text_series.isna() | (text_series.fillna("").astype(str).str.strip() == "")
            missing_text_indices = [int(idx) for idx in df.index[is_missing_text].tolist()]

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": int(len(df)),
            "missing_counts": missing_counts,
            "missing_text_count": len(missing_text_indices),
            "missing_text_indices_preview": missing_text_indices[:10]
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(
        subset: Optional[List[str]] = None,
        drop: bool = False
    ) -> Dict[str, Any]:
        """Checks for and optionally drops duplicate documents in the working dataset.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        When dropping (`drop=True`), uses `keep=False` to drop every copy of a duplicated row,
        matching Master notebook behavior (`X.drop_duplicates(keep=False, inplace=True)`).

        Args:
            subset: Optional list of column names to evaluate duplicates on (defaults to whole row if None).
            drop: Whether to drop duplicate rows from `session.dataframe` (default False).

        Returns:
            A summary dictionary containing n_duplicates, n_rows_involved, index preview,
            and before/after row counts, or an error dict if no dataset is loaded.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe

        # Helper to check if a column contains unhashable types (list, dict, set)
        def is_column_unhashable(col_name: str) -> bool:
            series = df[col_name]
            return series.apply(lambda val: isinstance(val, (list, dict, set))).any()

        unhashable_cols = [c for c in df.columns if is_column_unhashable(c)]

        if subset is not None:
            missing_cols = [c for c in subset if c not in df.columns]
            if missing_cols:
                return {"error": f"Columns specified in subset do not exist in dataset: {missing_cols}"}

            unhashable_in_subset = [c for c in subset if c in unhashable_cols]
            if unhashable_in_subset:
                return {"error": f"Cannot evaluate duplicates on unhashable column(s): {unhashable_in_subset}"}

            eval_subset = list(subset)
            excluded_cols = [c for c in df.columns if c not in eval_subset]
        else:
            eval_subset = [c for c in df.columns if c not in unhashable_cols]
            excluded_cols = unhashable_cols

        first_dup_mask = df.duplicated(subset=eval_subset, keep="first")
        n_duplicates = int(first_dup_mask.sum())

        all_dup_mask = df.duplicated(subset=eval_subset, keep=False)
        all_dup_indices = [int(idx) for idx in df.index[all_dup_mask].tolist()]
        n_rows_involved = len(all_dup_indices)

        rows_before = int(len(df))
        rows_after = rows_before
        dtm_stale_note = None

        if drop and n_rows_involved > 0:
            cleaned_df = df.drop_duplicates(subset=eval_subset, keep=False).reset_index(drop=True)
            session.dataframe = cleaned_df
            rows_after = int(len(cleaned_df))

            if session.feature_matrix is not None:
                dtm_stale_note = "session.dataframe was updated by dropping duplicates. Existing DTM feature_matrix is now stale; please re-run build_dtm_tool."

        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "status": "success",
            "subset": subset,
            "compared_columns": eval_subset,
            "excluded_columns": excluded_cols,
            "n_duplicates": n_duplicates,
            "n_rows_involved": n_rows_involved,
            "duplicate_indices_preview": all_dup_indices[:10],
            "rows_before": rows_before,
            "rows_after": rows_after,
            "dropped": drop
        }

        if dtm_stale_note:
            summary["note"] = dtm_stale_note

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"subset": subset, "drop": drop},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = 42,
        replace_working: bool = False
    ) -> Dict[str, Any]:
        """Subsamples the working dataset for preview or reduction.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        Returns a random sample of `n` documents. If `replace_working=True`, replaces `session.dataframe` with the sample.

        Args:
            n: Number of document rows to sample (default 5).
            random_state: Optional seed for reproducible sampling (default 42).
            replace_working: Whether to replace `session.dataframe` with the sampled subset (default False).

        Returns:
            A summary dictionary with sampled indices, per-category breakdown, and truncated row previews,
            or an error dict if parameters are invalid or no dataset is loaded.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe
        total_rows = len(df)

        if not isinstance(n, int) or n < 1 or n > total_rows:
            return {"error": f"n must be an integer between 1 and {total_rows} (got {n})."}

        sampled_df = df.sample(n=n, random_state=random_state)
        original_indices = [int(idx) for idx in sampled_df.index.tolist()]

        counts_per_category = {}
        if "category_name" in sampled_df.columns:
            counts_per_category = {str(cat): int(count) for cat, count in sampled_df["category_name"].value_counts().items()}

        sample_preview = []
        for idx, row in sampled_df.iterrows():
            text_val = str(row["text"]) if "text" in row and pd.notna(row["text"]) else ""
            cat_name = str(row["category_name"]) if "category_name" in row and pd.notna(row["category_name"]) else ""
            sample_preview.append({
                "index": int(idx),
                "category_name": cat_name,
                "text": text_val[:80]
            })

        dtm_stale_note = None
        if replace_working:
            session.dataframe = sampled_df.reset_index(drop=True)
            if session.feature_matrix is not None:
                dtm_stale_note = "session.dataframe was updated by sampling. Existing DTM feature_matrix is now stale; please re-run build_dtm_tool."

        result_id = session.next_result_id("sample_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_samples": n,
            "random_state": random_state,
            "sampled_indices": original_indices,
            "counts_per_category": counts_per_category,
            "sample_preview": sample_preview,
            "replaced_working": replace_working
        }

        if dtm_stale_note:
            summary["note"] = dtm_stale_note

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state, "replace_working": replace_working},
            summary=summary
        )

        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text into unigrams using NLTK word_tokenize.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        Lowercases text prior to tokenization. Stores resulting token lists in `session.dataframe["unigrams"]`.

        Returns:
            A summary dictionary containing document counts, token totals, average tokens per doc,
            and a token preview for the first document, or an error dict if no dataset is loaded.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe

        unigrams_list = []
        for text in df["text"]:
            text_str = "" if pd.isna(text) else str(text).lower()
            if not text_str.strip():
                unigrams_list.append([])
            else:
                try:
                    tokens = word_tokenize(text_str)
                except LookupError:
                    return {"error": "NLTK tokenizer resource missing. Please download using nltk.download('punkt') or nltk.download('punkt_tab')."}
                unigrams_list.append(tokens)

        session.dataframe["unigrams"] = unigrams_list

        total_tokens = sum(len(toks) for toks in unigrams_list)
        n_docs = len(df)
        avg_tokens = round(float(total_tokens / n_docs), 4) if n_docs > 0 else 0.0
        first_doc_preview = unigrams_list[0][:10] if n_docs > 0 else []

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "first_doc_preview": first_doc_preview
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on document text length and renders a box plot by category.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        Computes character length over `session.dataframe['text']` and calculates count, mean, std, min, 25%, 50%, 75%, max
        both overall and grouped by `category_name`. Builds a Seaborn boxplot and sets `session.pending_figure`.

        Returns:
            A summary dictionary containing overall_stats and category_stats, or an error dict if dataset or required columns are missing.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe

        if "text" not in df.columns or "category_name" not in df.columns:
            return {"error": "Required columns ('text' and 'category_name') are missing from dataset."}

        # Compute temporary text_length Series without mutating session.dataframe
        text_length = df["text"].fillna("").astype(str).apply(len)

        def series_to_stats(s: pd.Series) -> Dict[str, float]:
            desc = s.describe()
            return {
                "count": int(desc["count"]),
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4) if pd.notna(desc["std"]) else 0.0,
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"])
            }

        overall_stats = series_to_stats(text_length)

        category_stats = {}
        grouped = df.groupby("category_name")
        for cat_name, group in grouped:
            cat_len = text_length.loc[group.index]
            category_stats[str(cat_name)] = series_to_stats(cat_len)

        # Plot boxplot using temporary Series alongside category_name
        plot_df = pd.DataFrame({
            "category_name": df["category_name"],
            "text_length": text_length
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=plot_df, x="category_name", y="text_length", ax=ax)
        ax.set_title("Text Length Distribution by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "overall_stats": overall_stats,
            "category_stats": category_stats
        }

        # full_report combines overall and per-category stats into a clean DataFrame
        report_rows = [{"group": "overall", **overall_stats}]
        for cat, stats in category_stats.items():
            report_rows.append({"group": f"category_{cat}", **stats})
        full_report_df = pd.DataFrame(report_rows)

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        start: int = 0,
        max_chars: int = 100
    ) -> Dict[str, Any]:
        """Peeks at sequential rows of the working DataFrame for data inspection.

        `load_dataset_tool` must be run first to populate `session.dataframe`.
        Returns `n` consecutive rows starting at index `start` without random sampling.

        Args:
            n: Number of sequential rows to return (1 to 20, default 5).
            start: Zero-based starting row index (0 <= start < total_rows, default 0).
            max_chars: Maximum character length for text preview (20 to 500, default 100).

        Returns:
            A summary dictionary containing total_rows, column names, and row previews,
            or an error dict if parameters are out of range or no dataset is loaded.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe
        total_rows = len(df)

        if not isinstance(n, int) or n < 1 or n > 20:
            return {"error": f"n must be an integer between 1 and 20 (got {n})."}

        if not isinstance(start, int) or start < 0 or start >= total_rows:
            return {"error": f"start index must be an integer between 0 and {total_rows - 1} (got {start})."}

        if not isinstance(max_chars, int) or max_chars < 20 or max_chars > 500:
            return {"error": f"max_chars must be an integer between 20 and 500 (got {max_chars})."}

        sliced_df = df.iloc[start:start + n]
        columns_list = [str(c) for c in df.columns]

        rows_preview = []
        for idx, row in sliced_df.iterrows():
            orig_text = str(row["text"]) if "text" in row and pd.notna(row["text"]) else ""
            cat_name = str(row["category_name"]) if "category_name" in row and pd.notna(row["category_name"]) else ""
            rows_preview.append({
                "index": int(idx),
                "category_name": cat_name,
                "text": orig_text[:max_chars],
                "text_length": len(orig_text)
            })

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "columns": columns_list,
            "rows": rows_preview
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n, "start": start, "max_chars": max_chars},
            summary=summary
        )

        return summary

    return [
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool,
        inspect_data_tool
    ]
