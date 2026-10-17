import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes category labels in session.dataframe['category_name'].

        Returns:
            dict: Summary containing result_id, status, classes, shape [rows, cols],
            count per class, and the class to one-hot vector mapping.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session."}

        if session.dataframe.empty:
            return {"error": "DataFrame in session is empty."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' missing from DataFrame."}

        missing_count = session.dataframe["category_name"].isna().sum()
        if missing_count > 0:
            return {"error": f"Column 'category_name' has {missing_count} missing value(s)."}

        labels = session.dataframe["category_name"].astype(str).values
        classes = sorted(np.unique(labels).tolist())

        # Directly compute the (n_documents, n_classes) one-hot matrix
        binary_matrix = (labels[:, None] == np.array(classes)[None, :]).astype(int)

        # Store in session.artifacts
        session.artifacts["label_matrix"] = binary_matrix
        session.artifacts["label_classes"] = classes

        # Pre-compute count per class
        unique_labels, counts = np.unique(labels, return_counts=True)
        counts_per_class = {str(k): int(v) for k, v in zip(unique_labels, counts)}

        # Construct class-to-vector mapping (n_classes entries)
        identity = np.eye(len(classes), dtype=int)
        class_to_vec = {cls_name: identity[i].tolist() for i, cls_name in enumerate(classes)}

        res_id = session.next_result_id("binarize")
        summary = {
            "result_id": res_id,
            "status": "success",
            "classes": classes,
            "shape": list(binary_matrix.shape),
            "counts_per_class": counts_per_class,
            "class_to_vector": class_to_vec
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> dict:
        """Reduces document-term matrix dimensions (PCA, t-SNE, UMAP) and generates a scatter plot.

        Args:
            method: Dimensionality reduction algorithm ('pca', 'tsne', or 'umap').
            n_components: Target number of dimensions (2 or 3).
            random_state: Seed for reproducible output.
            perplexity: Perplexity parameter for t-SNE (must be < n_samples).
            n_neighbors: Number of neighbors parameter for UMAP (must be < n_samples).

        Returns:
            dict: Summary containing result_id, method, params, shape,
            explained_variance_ratio (PCA only), and preview of first 5 points.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found in session. Run build_dtm_tool first."}

        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' missing from DataFrame."}

        n_samples = session.feature_matrix.shape[0]
        if len(session.dataframe) != n_samples:
            return {"error": "DataFrame row count does not match feature_matrix shape. DTM is stale."}

        if n_components not in [2, 3]:
            return {"error": f"n_components must be 2 or 3, got {n_components}."}

        # Parameter validation for t-SNE and UMAP
        if method == "tsne" and perplexity >= n_samples:
            return {"error": f"perplexity ({perplexity}) must be less than n_samples ({n_samples})."}

        if method == "umap" and n_neighbors >= n_samples:
            return {"error": f"n_neighbors ({n_neighbors}) must be less than n_samples ({n_samples})."}

        # Memory guard before densifying
        n_rows, n_cols = session.feature_matrix.shape
        if n_rows * n_cols > 20_000_000:
            return {"error": f"Matrix total elements ({n_rows * n_cols}) exceeds 20,000,000 limit. Rebuild DTM with build_dtm_tool(max_features=...)."}

        # Densify feature matrix
        if scipy.sparse.issparse(session.feature_matrix):
            X = session.feature_matrix.toarray()
        else:
            X = np.asarray(session.feature_matrix)

        explained_variance_ratio = None

        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(X)
            explained_variance_ratio = [float(np.round(x, 4)) for x in pca.explained_variance_ratio_]

        elif method == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = tsne.fit_transform(X)

        elif method == "umap":
            try:
                import umap
            except ImportError:
                return {"error": "umap-learn package is not installed."}

            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)

        else:
            return {"error": f"Unknown method '{method}'. Must be 'pca', 'tsne', or 'umap'."}

        coords = np.asarray(coords)
        session.artifacts[f"reduced_{method}"] = coords

        # Generate 2D scatter plot colored by category_name using positional numpy array
        fig, ax = plt.subplots(figsize=(8, 6))
        category_labels = session.dataframe["category_name"].astype(str).to_numpy()

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=category_labels,
            style=category_labels,
            ax=ax,
            s=50
        )

        ax.set_title(f"{method.upper()} Dimensionality Reduction")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.tight_layout()

        session.pending_figure = fig

        res_id = session.next_result_id("reduce_dims")
        preview = [[float(np.round(val, 4)) for val in row] for row in coords[:5]]

        tool_args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors
        }

        summary = {
            "result_id": res_id,
            "status": "success",
            "method": method,
            "params": tool_args,
            "shape": list(coords.shape),
            "explained_variance_ratio": explained_variance_ratio,
            "coordinates_preview": preview
        }

        session.store_result("reduce_dimensions_tool", tool_args, summary)
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
