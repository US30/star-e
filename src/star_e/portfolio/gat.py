"""
Graph Attention Networks for asset correlation clustering.

Uses GAT to learn representations of assets based on their correlation
structure, enabling better portfolio clustering and risk assessment.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import networkx as nx
from sklearn.cluster import SpectralClustering
from typing import Optional, List, Tuple, Dict
import mlflow


class AssetGATEncoder(nn.Module):
    """
    Graph Attention Network encoder for asset embeddings.

    Learns node embeddings that capture the correlation structure
    between assets using attention mechanisms.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            out_dim = embedding_dim if i == num_layers - 1 else hidden_dim

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=i < num_layers - 1,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim * (num_heads if i < num_layers - 1 else 1)))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Node features (n_nodes, in_features)
            edge_index: Edge connectivity (2, n_edges)
            edge_attr: Optional edge features

        Returns:
            Node embeddings and attention weights per layer
        """
        attention_weights = []

        x = self.input_proj(x)
        x = F.elu(x)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x, attn = gat(x, edge_index, return_attention_weights=True)
            attention_weights.append(attn)
            x = norm(x)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x, attention_weights


class AssetCorrelationGraph:
    """
    Builds and manages the asset correlation graph.

    Creates a graph where nodes are assets and edges represent
    correlations above a threshold.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        use_absolute: bool = True,
        top_k_edges: Optional[int] = None,
    ):
        self.correlation_threshold = correlation_threshold
        self.use_absolute = use_absolute
        self.top_k_edges = top_k_edges

        self.adj_matrix: Optional[np.ndarray] = None
        self.tickers: Optional[List[str]] = None
        self.nx_graph: Optional[nx.Graph] = None

    def build_from_returns(
        self,
        returns: pd.DataFrame,
        method: str = "pearson",
    ) -> "AssetCorrelationGraph":
        """
        Build correlation graph from return series.

        Args:
            returns: DataFrame with asset returns (columns are assets)
            method: Correlation method ("pearson", "spearman", "kendall")

        Returns:
            Self for chaining
        """
        self.tickers = list(returns.columns)

        corr_matrix = returns.corr(method=method).values

        if self.use_absolute:
            corr_matrix = np.abs(corr_matrix)

        adj_matrix = np.where(corr_matrix >= self.correlation_threshold, corr_matrix, 0)
        np.fill_diagonal(adj_matrix, 0)

        if self.top_k_edges is not None:
            flat_idx = np.argsort(adj_matrix.flatten())[::-1]
            mask = np.zeros_like(adj_matrix, dtype=bool).flatten()
            mask[flat_idx[:self.top_k_edges * 2]] = True
            mask = mask.reshape(adj_matrix.shape)
            adj_matrix = np.where(mask, adj_matrix, 0)
            adj_matrix = (adj_matrix + adj_matrix.T) / 2

        self.adj_matrix = adj_matrix

        self.nx_graph = nx.from_numpy_array(adj_matrix)
        nx.relabel_nodes(
            self.nx_graph,
            {i: self.tickers[i] for i in range(len(self.tickers))},
            copy=False,
        )

        return self

    def to_pyg_data(
        self,
        node_features: Optional[np.ndarray] = None,
    ) -> Data:
        """
        Convert to PyTorch Geometric Data object.

        Args:
            node_features: Optional node feature matrix

        Returns:
            PyG Data object
        """
        if self.adj_matrix is None:
            raise ValueError("Graph must be built first")

        adj_tensor = torch.FloatTensor(self.adj_matrix)
        edge_index, edge_attr = dense_to_sparse(adj_tensor)

        if node_features is None:
            x = torch.eye(len(self.tickers))
        else:
            x = torch.FloatTensor(node_features)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.tickers),
        )

    def get_centrality_measures(self) -> pd.DataFrame:
        """Calculate various centrality measures for each asset."""
        if self.nx_graph is None:
            raise ValueError("Graph must be built first")

        degree = dict(self.nx_graph.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(self.nx_graph, weight="weight")
        closeness = nx.closeness_centrality(self.nx_graph, distance="weight")

        try:
            eigenvector = nx.eigenvector_centrality(
                self.nx_graph, max_iter=500, weight="weight"
            )
        except nx.PowerIterationFailedConvergence:
            eigenvector = {node: 0 for node in self.nx_graph.nodes()}

        return pd.DataFrame({
            "ticker": list(degree.keys()),
            "degree_centrality": list(degree.values()),
            "betweenness_centrality": list(betweenness.values()),
            "closeness_centrality": list(closeness.values()),
            "eigenvector_centrality": list(eigenvector.values()),
        })

    def get_communities(
        self,
        method: str = "louvain",
    ) -> Dict[str, int]:
        """
        Detect communities in the correlation graph.

        Args:
            method: Community detection method

        Returns:
            Dictionary mapping ticker to community ID
        """
        if self.nx_graph is None:
            raise ValueError("Graph must be built first")

        if method == "louvain":
            import community as community_louvain
            partition = community_louvain.best_partition(self.nx_graph, weight="weight")
            return partition
        elif method == "spectral":
            n_clusters = min(5, len(self.tickers))
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
            )
            labels = clustering.fit_predict(self.adj_matrix + 1e-6)
            return {self.tickers[i]: int(labels[i]) for i in range(len(self.tickers))}
        else:
            raise ValueError(f"Unknown method: {method}")


class GATClusterer:
    """
    Uses GAT to cluster assets based on learned embeddings.

    Combines graph structure learning with clustering for
    portfolio construction.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        device: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model: Optional[AssetGATEncoder] = None
        self.graph: Optional[AssetCorrelationGraph] = None
        self.embeddings: Optional[np.ndarray] = None

    def fit(
        self,
        returns: pd.DataFrame,
        node_features: Optional[np.ndarray] = None,
        correlation_threshold: float = 0.3,
    ) -> "GATClusterer":
        """
        Fit GAT on asset correlation graph.

        Uses reconstruction loss to learn meaningful embeddings.

        Args:
            returns: Asset returns DataFrame
            node_features: Optional node features
            correlation_threshold: Threshold for graph construction

        Returns:
            Self for chaining
        """
        self.graph = AssetCorrelationGraph(correlation_threshold=correlation_threshold)
        self.graph.build_from_returns(returns)

        if node_features is None:
            node_features = self._compute_default_features(returns)

        data = self.graph.to_pyg_data(node_features)
        data = data.to(self.device)

        n_nodes = data.num_nodes
        in_features = node_features.shape[1]

        self.model = AssetGATEncoder(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        ).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            decoder.train()
            optimizer.zero_grad()

            embeddings, _ = self.model(data.x, data.edge_index)

            reconstructed = torch.mm(embeddings, embeddings.t())
            reconstructed = torch.sigmoid(reconstructed)

            target = to_dense_adj(data.edge_index, max_num_nodes=n_nodes).squeeze()

            loss = F.binary_cross_entropy(reconstructed, target)

            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            self.embeddings, _ = self.model(data.x, data.edge_index)
            self.embeddings = self.embeddings.cpu().numpy()

        mlflow.log_params({
            "gat_hidden_dim": self.hidden_dim,
            "gat_embedding_dim": self.embedding_dim,
            "gat_num_heads": self.num_heads,
            "gat_epochs": self.epochs,
        })

        return self

    def _compute_default_features(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute default node features from returns."""
        features = pd.DataFrame()

        features["mean_return"] = returns.mean()
        features["volatility"] = returns.std()
        features["skewness"] = returns.skew()
        features["kurtosis"] = returns.kurtosis()
        features["sharpe"] = returns.mean() / returns.std()

        for window in [5, 10, 21]:
            rolling_vol = returns.rolling(window).std().iloc[-1]
            features[f"vol_{window}d"] = rolling_vol

        return features.values

    def get_embeddings(self) -> pd.DataFrame:
        """Get learned asset embeddings."""
        if self.embeddings is None or self.graph is None:
            raise ValueError("Model must be fitted first")

        return pd.DataFrame(
            self.embeddings,
            index=self.graph.tickers,
            columns=[f"emb_{i}" for i in range(self.embedding_dim)],
        )

    def cluster_assets(
        self,
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> Dict[str, int]:
        """
        Cluster assets based on learned embeddings.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ("kmeans", "spectral", "agglomerative")

        Returns:
            Dictionary mapping ticker to cluster ID
        """
        if self.embeddings is None:
            raise ValueError("Model must be fitted first")

        if method == "kmeans":
            from sklearn.cluster import KMeans
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "spectral":
            clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")

        labels = clustering.fit_predict(self.embeddings)

        return {
            self.graph.tickers[i]: int(labels[i])
            for i in range(len(self.graph.tickers))
        }

    def get_attention_matrix(
        self,
        returns: pd.DataFrame,
        node_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get attention weights matrix from GAT.

        Reveals which asset pairs the model focuses on.
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model must be fitted first")

        if node_features is None:
            node_features = self._compute_default_features(returns)

        data = self.graph.to_pyg_data(node_features)
        data = data.to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(data.x, data.edge_index)

        last_attn = attention_weights[-1]
        edge_index = last_attn[0].cpu().numpy()
        attn_values = last_attn[1].cpu().numpy().mean(axis=1)

        n_nodes = len(self.graph.tickers)
        attn_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            attn_matrix[src, dst] = attn_values[i]

        return attn_matrix


def cluster_portfolio_with_gat(
    returns: pd.DataFrame,
    n_clusters: int = 5,
    correlation_threshold: float = 0.3,
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Convenience function to cluster portfolio assets using GAT.

    Args:
        returns: Asset returns DataFrame
        n_clusters: Number of clusters
        correlation_threshold: Graph construction threshold

    Returns:
        Tuple of (cluster assignments, embeddings)
    """
    clusterer = GATClusterer()
    clusterer.fit(returns, correlation_threshold=correlation_threshold)

    clusters = clusterer.cluster_assets(n_clusters=n_clusters)
    embeddings = clusterer.get_embeddings()

    return clusters, embeddings.values
