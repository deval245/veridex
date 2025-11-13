"""Visualization tools for cultural embeddings."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    country_names: List[str],
    save_path: Optional[Path] = None,
    top_n: int = 30
):
    """Plot country similarity heatmap."""
    # Select top N most frequent countries
    indices = list(range(min(top_n, len(country_names))))
    sub_matrix = similarity_matrix[np.ix_(indices, indices)]
    sub_names = [country_names[i] for i in indices]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sub_matrix,
        xticklabels=sub_names,
        yticklabels=sub_names,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        annot=False,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f'Cultural Similarity Matrix (Top {top_n} Countries)', fontsize=16, pad=20)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_cultural_space(
    embeddings_3d: np.ndarray,
    country_names: List[str],
    save_path: Optional[Path] = None
):
    """Interactive 3D scatter plot of cultural embeddings."""
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color=embeddings_3d[:, 0],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="PC1")
        ),
        text=country_names,
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Cultural Embedding Space (PCA Projection)',
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        ),
        width=1000,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_tsne_2d(
    embeddings_2d: np.ndarray,
    country_names: List[str],
    save_path: Optional[Path] = None
):
    """2D t-SNE visualization."""
    plt.figure(figsize=(16, 12))
    
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6, c=range(len(country_names)), cmap='tab20')
    
    for i, name in enumerate(country_names):
        plt.annotate(
            name,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('Cultural Embedding Space (t-SNE Projection)', fontsize=16, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dimension_importance(
    explained_variance: np.ndarray,
    save_path: Optional[Path] = None
):
    """Plot PCA explained variance."""
    plt.figure(figsize=(10, 6))
    
    dims = list(range(1, len(explained_variance) + 1))
    cumulative = np.cumsum(explained_variance)
    
    plt.bar(dims, explained_variance, alpha=0.6, label='Individual')
    plt.plot(dims, cumulative, 'ro-', label='Cumulative', linewidth=2)
    
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.title('Cultural Embedding Dimension Importance', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    country_names: List[str],
    save_path: Optional[Path] = None
):
    """Hierarchical clustering dendrogram."""
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    
    plt.figure(figsize=(16, 10))
    
    scipy_dendrogram(
        linkage_matrix,
        labels=country_names,
        leaf_font_size=9,
        leaf_rotation=90
    )
    
    plt.title('Hierarchical Clustering of Cultural Policies', fontsize=16, pad=20)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_visualizations(analysis_dir: Path, viz_dir: Path, country_mapping: Dict[int, str]):
    """Generate all visualization outputs."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    country_names = [country_mapping[i] for i in range(len(country_mapping))]
    
    # Load analysis results
    sim_matrix = np.load(analysis_dir / 'similarity_matrix.npy')
    pca_3d = np.load(analysis_dir / 'pca_projections.npy')
    pca_var = np.load(analysis_dir / 'pca_explained_variance.npy')
    tsne_2d = np.load(analysis_dir / 'tsne_2d.npy')
    linkage_matrix = np.load(analysis_dir / 'linkage_matrix.npy')
    
    # Generate visualizations
    plot_similarity_heatmap(sim_matrix, country_names, viz_dir / 'similarity_heatmap.png')
    plot_3d_cultural_space(pca_3d, country_names, viz_dir / 'cultural_space_3d.html')
    plot_tsne_2d(tsne_2d, country_names, viz_dir / 'tsne_2d.png')
    plot_dimension_importance(pca_var, viz_dir / 'dimension_importance.png')
    plot_dendrogram(linkage_matrix, country_names, viz_dir / 'dendrogram.png')
