"""Post-training analysis script for cultural embeddings."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.architectures.veridex_cultural import VERIDEXCultural
from src.evaluation.cultural_analysis import run_analysis
from src.evaluation.visualize import create_all_visualizations
from src.constants.countries import get_manager


def main():
    """Run complete cultural embedding analysis."""
    
    # Paths
    checkpoint_path = Path('/content/drive/MyDrive/veridex_cultural_embeddings/best_model.pt')
    output_dir = Path('/content/drive/MyDrive/veridex_cultural_embeddings/analysis')
    viz_dir = Path('/content/drive/MyDrive/veridex_cultural_embeddings/visualizations')
    
    print("═" * 70)
    print("CULTURAL EMBEDDING ANALYSIS")
    print("═" * 70)
    print()
    
    # Load country mapping
    print("[1/5] Loading country mapping...")
    country_manager = get_manager()
    country_mapping = {idx: country_manager.get_country_code(idx) for idx in range(country_manager.get_num_countries())}
    print(f"      {len(country_mapping)} countries loaded")
    print()
    
    # Load model
    print("[2/5] Loading trained model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = VERIDEXCultural(
        model_name="microsoft/deberta-v3-base",
        num_countries=len(country_mapping),
        num_classes=checkpoint.get('num_classes', 51),
        cultural_dim=8,
        dropout=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("      Model loaded successfully")
    print()
    
    # Run analysis
    print("[3/5] Analyzing cultural embeddings...")
    analyzer = run_analysis(model, country_mapping, output_dir)
    print("      Analysis complete:")
    print(f"      - Similarity matrix: {output_dir / 'similarity_matrix.npy'}")
    print(f"      - PCA projections: {output_dir / 'pca_projections.npy'}")
    print(f"      - t-SNE 2D: {output_dir / 'tsne_2d.npy'}")
    print(f"      - Dimension analysis: {output_dir / 'dimension_analysis.json'}")
    print(f"      - Nearest neighbors: {output_dir / 'nearest_neighbors.json'}")
    print()
    
    # Create visualizations
    print("[4/5] Generating visualizations...")
    create_all_visualizations(output_dir, viz_dir, country_mapping)
    print("      Visualizations saved:")
    print(f"      - Similarity heatmap: {viz_dir / 'similarity_heatmap.png'}")
    print(f"      - 3D cultural space: {viz_dir / 'cultural_space_3d.html'}")
    print(f"      - t-SNE 2D: {viz_dir / 'tsne_2d.png'}")
    print(f"      - Dimension importance: {viz_dir / 'dimension_importance.png'}")
    print(f"      - Dendrogram: {viz_dir / 'dendrogram.png'}")
    print()
    
    # Summary statistics
    print("[5/5] Computing summary statistics...")
    sim_matrix = analyzer.compute_similarity_matrix()
    
    # Most similar country pairs
    upper_tri_indices = torch.triu_indices(sim_matrix.shape[0], sim_matrix.shape[0], offset=1)
    upper_tri_sims = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]
    top_pairs_idx = torch.topk(torch.from_numpy(upper_tri_sims), k=10).indices
    
    print("      Top 10 most similar country pairs:")
    for idx in top_pairs_idx:
        i, j = upper_tri_indices[0][idx].item(), upper_tri_indices[1][idx].item()
        sim = sim_matrix[i, j]
        print(f"        {country_mapping[i]:3} - {country_mapping[j]:3}: {sim:.4f}")
    print()
    
    print("═" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("═" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations: {viz_dir}")


if __name__ == "__main__":
    main()

