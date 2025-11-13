"""Cultural embedding analysis and interpretation."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns


class CulturalAnalyzer:
    """Extract insights from learned cultural embeddings."""
    
    def __init__(self, model_path: Path, country_mapping: Dict[int, str]):
        self.model_path = model_path
        self.country_mapping = country_mapping
        self.embeddings = None
        self.num_countries = len(country_mapping)
        
    def load_embeddings(self, model):
        """Extract cultural embeddings from trained model."""
        cultural_layer = model.cultural_encoder.cultural_embedding
        with torch.no_grad():
            all_ids = torch.arange(self.num_countries)
            self.embeddings = cultural_layer(all_ids).cpu().numpy()
        return self.embeddings
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        n = self.num_countries
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = 1 - cosine(self.embeddings[i], self.embeddings[j])
        
        return sim_matrix
    
    def find_nearest_neighbors(self, country_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Find k most similar countries."""
        query_emb = self.embeddings[country_id]
        
        similarities = []
        for i in range(self.num_countries):
            if i != country_id:
                sim = 1 - cosine(query_emb, self.embeddings[i])
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]
    
    def cluster_countries(self, method: str = 'ward') -> np.ndarray:
        """Hierarchical clustering of countries."""
        linkage_matrix = linkage(self.embeddings, method=method)
        return linkage_matrix
    
    def pca_analysis(self, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """PCA dimensionality reduction."""
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.embeddings)
        explained_var = pca.explained_variance_ratio_
        return reduced, explained_var
    
    def tsne_projection(self, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
        """t-SNE projection for visualization."""
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        projected = tsne.fit_transform(self.embeddings)
        return projected
    
    def analyze_dimensions(self) -> Dict[int, Dict]:
        """Analyze what each embedding dimension captures."""
        analysis = {}
        
        for dim in range(self.embeddings.shape[1]):
            values = self.embeddings[:, dim]
            
            highest_ids = np.argsort(values)[-5:][::-1]
            lowest_ids = np.argsort(values)[:5]
            
            analysis[dim] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'range': (float(np.min(values)), float(np.max(values))),
                'highest_countries': [(int(i), self.country_mapping[i], float(values[i])) for i in highest_ids],
                'lowest_countries': [(int(i), self.country_mapping[i], float(values[i])) for i in lowest_ids]
            }
        
        return analysis
    
    def export_results(self, output_dir: Path):
        """Export all analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Similarity matrix
        sim_matrix = self.compute_similarity_matrix()
        np.save(output_dir / 'similarity_matrix.npy', sim_matrix)
        
        # Clustering
        linkage_matrix = self.cluster_countries()
        np.save(output_dir / 'linkage_matrix.npy', linkage_matrix)
        
        # PCA
        pca_reduced, pca_var = self.pca_analysis()
        np.save(output_dir / 'pca_projections.npy', pca_reduced)
        np.save(output_dir / 'pca_explained_variance.npy', pca_var)
        
        # t-SNE
        tsne_2d = self.tsne_projection(n_components=2)
        np.save(output_dir / 'tsne_2d.npy', tsne_2d)
        
        # Dimension analysis
        dim_analysis = self.analyze_dimensions()
        with open(output_dir / 'dimension_analysis.json', 'w') as f:
            json.dump(dim_analysis, f, indent=2)
        
        # Nearest neighbors for all countries
        neighbors_data = {}
        for country_id in range(self.num_countries):
            neighbors = self.find_nearest_neighbors(country_id, k=5)
            neighbors_data[self.country_mapping[country_id]] = [
                {'country': self.country_mapping[nid], 'similarity': float(sim)}
                for nid, sim in neighbors
            ]
        
        with open(output_dir / 'nearest_neighbors.json', 'w') as f:
            json.dump(neighbors_data, f, indent=2)


def run_analysis(model, country_mapping: Dict[int, str], output_dir: Path):
    """Run complete cultural embedding analysis."""
    analyzer = CulturalAnalyzer(None, country_mapping)
    analyzer.load_embeddings(model)
    analyzer.export_results(output_dir)
    
    return analyzer

