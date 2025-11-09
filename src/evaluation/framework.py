import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from scipy import stats as scipy_stats
from collections import defaultdict


@dataclass
class EvaluationMetrics:
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    coverage: float
    avg_confidence: float
    total_predictions: int
    correct_predictions: int
    

class EvaluationFramework:
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_predictions(
        self,
        predictions: List[Any],
        ground_truth: Dict[str, str],
        method_name: str
    ) -> EvaluationMetrics:
        
        correct = 0
        total_with_ground_truth = 0
        total_predictions = len(predictions)
        
        tp = fp = fn = tn = 0
        confidences = []
        
        for pred in predictions:
            key = f"{pred.content_id}_{pred.region}"
            true_rating = ground_truth.get(key)
            
            if pred.predicted_rating:
                confidences.append(pred.confidence)
            
            if true_rating:
                total_with_ground_truth += 1
                
                if pred.predicted_rating == true_rating:
                    correct += 1
                    tp += 1
                else:
                    if pred.predicted_rating:
                        fp += 1
                    else:
                        fn += 1
        
        accuracy = correct / total_with_ground_truth if total_with_ground_truth > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        coverage = total_with_ground_truth / total_predictions if total_predictions > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return EvaluationMetrics(
            method=method_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            coverage=coverage,
            avg_confidence=avg_confidence,
            total_predictions=total_predictions,
            correct_predictions=correct
        )
    
    def compare_methods(
        self,
        methods_metrics: List[EvaluationMetrics]
    ) -> Dict[str, Any]:
        
        comparison = {
            "methods": {},
            "best_by_metric": {}
        }
        
        for metrics in methods_metrics:
            comparison["methods"][metrics.method] = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "coverage": metrics.coverage,
                "confidence": metrics.avg_confidence
            }
        
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            best = max(methods_metrics, key=lambda m: getattr(m, metric_name))
            comparison["best_by_metric"][metric_name] = {
                "method": best.method,
                "value": getattr(best, metric_name)
            }
        
        return comparison
    
    def statistical_significance_test(
        self,
        method1_predictions: List[bool],
        method2_predictions: List[bool]
    ) -> Dict[str, Any]:
        
        if len(method1_predictions) != len(method2_predictions):
            return {"error": "Different number of predictions"}
        
        acc1 = sum(method1_predictions) / len(method1_predictions)
        acc2 = sum(method2_predictions) / len(method2_predictions)
        
        contingency_table = [
            [sum(method1_predictions), len(method1_predictions) - sum(method1_predictions)],
            [sum(method2_predictions), len(method2_predictions) - sum(method2_predictions)]
        ]
        
        chi2, p_value = scipy_stats.chi2_contingency(contingency_table)[:2]
        
        return {
            "method1_accuracy": acc1,
            "method2_accuracy": acc2,
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01
        }
    
    def compute_confusion_matrix(
        self,
        predictions: List[Any],
        ground_truth: Dict[str, str],
        rating_categories: List[str]
    ) -> Dict[str, Any]:
        
        matrix = defaultdict(lambda: defaultdict(int))
        
        for pred in predictions:
            key = f"{pred.content_id}_{pred.region}"
            true_rating = ground_truth.get(key)
            
            if true_rating and pred.predicted_rating:
                matrix[true_rating][pred.predicted_rating] += 1
        
        return {
            "matrix": {k: dict(v) for k, v in matrix.items()},
            "rating_categories": rating_categories
        }
    
    def analyze_by_region(
        self,
        predictions: List[Any],
        ground_truth: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        
        by_region = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred in predictions:
            key = f"{pred.content_id}_{pred.region}"
            true_rating = ground_truth.get(key)
            
            if true_rating:
                by_region[pred.region]["total"] += 1
                if pred.predicted_rating == true_rating:
                    by_region[pred.region]["correct"] += 1
        
        return {
            region: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "total": stats["total"]
            }
            for region, stats in by_region.items()
        }
    
    def generate_report(
        self,
        all_metrics: List[EvaluationMetrics],
        comparison: Dict[str, Any],
        output_name: str = "evaluation_report.json"
    ) -> Path:
        
        report_path = self.output_dir / output_name
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "methods_evaluated": len(all_metrics),
            "metrics": [
                {
                    "method": m.method,
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "coverage": m.coverage,
                    "confidence": m.avg_confidence,
                    "total_predictions": m.total_predictions,
                    "correct_predictions": m.correct_predictions
                }
                for m in all_metrics
            ],
            "comparison": comparison
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def print_comparison_table(self, metrics_list: List[EvaluationMetrics]):
        
        print("\n" + "=" * 100)
        print("METHOD COMPARISON")
        print("=" * 100)
        print(f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Coverage':<12}")
        print("-" * 100)
        
        for m in sorted(metrics_list, key=lambda x: x.accuracy, reverse=True):
            print(f"{m.method:<20} {m.accuracy:>10.1%}  {m.precision:>10.1%}  {m.recall:>10.1%}  {m.f1:>10.1%}  {m.coverage:>10.1%}")
        
        print("=" * 100 + "\n")


from datetime import datetime

