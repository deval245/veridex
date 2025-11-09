import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


class VisualizationGenerator:
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(
        self,
        metrics: List[Any],
        by_region: Dict[str, Dict],
        confusion_matrices: Optional[Dict] = None
    ) -> Dict[str, Path]:
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.5)
            
            figures = {}
            
            figures['accuracy_comparison'] = self._plot_accuracy_comparison(metrics, plt, sns)
            figures['regional_performance'] = self._plot_regional_performance(by_region, plt, sns)
            figures['metrics_radar'] = self._plot_metrics_radar(metrics, plt, sns)
            
            if confusion_matrices:
                figures['confusion_matrix'] = self._plot_confusion_matrix(
                    confusion_matrices, plt, sns
                )
            
            plt.close('all')
            
            return figures
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not installed. Skipping visualizations.")
            return {}
    
    def _plot_accuracy_comparison(self, metrics: List, plt, sns) -> Path:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = [m.method for m in metrics]
        accuracies = [m.accuracy * 100 for m in metrics]
        
        colors = sns.color_palette("husl", len(methods))
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_title('Accuracy Comparison Across Methods', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "accuracy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_regional_performance(self, by_region: Dict, plt, sns) -> Path:
        
        fig, axes = plt.subplots(1, len(by_region), figsize=(15, 6), sharey=True)
        
        if len(by_region) == 1:
            axes = [axes]
        
        for idx, (method, regions) in enumerate(by_region.items()):
            ax = axes[idx]
            
            sorted_regions = sorted(regions.items(), 
                                   key=lambda x: x[1]['accuracy'], 
                                   reverse=True)
            
            region_names = [r[0] for r in sorted_regions]
            accuracies = [r[1]['accuracy'] * 100 for r in sorted_regions]
            
            colors = sns.color_palette("viridis", len(region_names))
            bars = ax.barh(region_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
            
            for bar, acc in zip(bars, accuracies):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{acc:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(method.upper(), fontsize=14, fontweight='bold')
            ax.set_xlim(0, 110)
            ax.grid(axis='x', alpha=0.3)
        
        fig.suptitle('Performance by Region', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / "regional_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_metrics_radar(self, metrics: List, plt, sns) -> Path:
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Coverage']
        num_vars = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        colors = sns.color_palette("Set2", len(metrics))
        
        for idx, m in enumerate(metrics):
            values = [m.accuracy, m.precision, m.recall, m.f1, m.coverage]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=m.method.upper(), 
                   color=colors[idx], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.set_title('Multi-Metric Comparison', fontsize=16, fontweight='bold', pad=40)
        
        plt.tight_layout()
        output_path = self.output_dir / "metrics_radar.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_confusion_matrix(self, confusion_data: Dict, plt, sns) -> Path:
        
        matrix = confusion_data.get('matrix', {})
        categories = confusion_data.get('rating_categories', [])
        
        if not matrix or not categories:
            return None
        
        matrix_array = np.zeros((len(categories), len(categories)))
        for i, true_label in enumerate(categories):
            for j, pred_label in enumerate(categories):
                matrix_array[i, j] = matrix.get(true_label, {}).get(pred_label, 0)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(matrix_array, annot=True, fmt='g', cmap='Blues',
                   xticklabels=categories, yticklabels=categories,
                   cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
        
        ax.set_xlabel('Predicted Rating', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Rating', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_latex_table(
        self,
        metrics: List[Any],
        output_name: str = "results_table.tex"
    ) -> Path:
        
        output_path = self.output_dir / output_name
        
        latex = r"""\begin{table}[h]
\centering
\caption{Performance Comparison of Content Rating Validation Methods}
\label{tab:results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Coverage} & \textbf{Confidence} \\
\midrule
"""
        
        for m in sorted(metrics, key=lambda x: x.accuracy, reverse=True):
            latex += f"{m.method.upper()} & {m.accuracy:.3f} & {m.precision:.3f} & {m.recall:.3f} & {m.f1:.3f} & {m.coverage:.3f} & {m.avg_confidence:.3f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(output_path, 'w') as f:
            f.write(latex)
        
        return output_path

