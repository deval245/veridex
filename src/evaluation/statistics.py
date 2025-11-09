import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from dataclasses import dataclass


@dataclass
class SignificanceTest:
    test_name: str
    statistic: float
    p_value: float
    significant_at_005: bool
    significant_at_001: bool
    interpretation: str


class StatisticalTester:
    
    @staticmethod
    def mcnemar_test(
        method1_correct: List[bool],
        method2_correct: List[bool]
    ) -> SignificanceTest:
        
        n = len(method1_correct)
        
        b = sum(1 for i in range(n) if method1_correct[i] and not method2_correct[i])
        c = sum(1 for i in range(n) if not method1_correct[i] and method2_correct[i])
        
        if (b + c) == 0:
            return SignificanceTest(
                test_name="McNemar",
                statistic=0.0,
                p_value=1.0,
                significant_at_005=False,
                significant_at_001=False,
                interpretation="No disagreements between methods"
            )
        
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        interpretation = f"Method 1 correct {b} times when Method 2 wrong, Method 2 correct {c} times when Method 1 wrong"
        
        return SignificanceTest(
            test_name="McNemar",
            statistic=statistic,
            p_value=p_value,
            significant_at_005=p_value < 0.05,
            significant_at_001=p_value < 0.01,
            interpretation=interpretation
        )
    
    @staticmethod
    def paired_t_test(
        method1_scores: List[float],
        method2_scores: List[float]
    ) -> SignificanceTest:
        
        statistic, p_value = stats.ttest_rel(method1_scores, method2_scores)
        
        mean_diff = np.mean(method1_scores) - np.mean(method2_scores)
        interpretation = f"Mean difference: {mean_diff:.4f}"
        
        return SignificanceTest(
            test_name="Paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant_at_005=p_value < 0.05,
            significant_at_001=p_value < 0.01,
            interpretation=interpretation
        )
    
    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        
        scores_array = np.array(scores)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return {
            "mean": float(np.mean(scores)),
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": confidence
        }
    
    @staticmethod
    def effect_size_cohens_d(
        method1_scores: List[float],
        method2_scores: List[float]
    ) -> float:
        
        mean1, mean2 = np.mean(method1_scores), np.mean(method2_scores)
        std1, std2 = np.std(method1_scores, ddof=1), np.std(method2_scores, ddof=1)
        
        n1, n2 = len(method1_scores), len(method2_scores)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        return float(cohens_d)
    
    def comprehensive_comparison(
        self,
        method1_results: List[bool],
        method2_results: List[bool],
        method1_scores: List[float],
        method2_scores: List[float],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2"
    ) -> Dict[str, Any]:
        
        mcnemar = self.mcnemar_test(method1_results, method2_results)
        
        t_test = self.paired_t_test(method1_scores, method2_scores)
        
        ci1 = self.bootstrap_confidence_interval(method1_scores)
        ci2 = self.bootstrap_confidence_interval(method2_scores)
        
        effect_size = self.effect_size_cohens_d(method1_scores, method2_scores)
        
        return {
            "method1_name": method1_name,
            "method2_name": method2_name,
            "mcnemar_test": {
                "statistic": mcnemar.statistic,
                "p_value": mcnemar.p_value,
                "significant_at_0.05": mcnemar.significant_at_005,
                "significant_at_0.01": mcnemar.significant_at_001,
                "interpretation": mcnemar.interpretation
            },
            "paired_t_test": {
                "statistic": t_test.statistic,
                "p_value": t_test.p_value,
                "significant_at_0.05": t_test.significant_at_005,
                "significant_at_0.01": t_test.significant_at_001
            },
            "confidence_intervals": {
                method1_name: ci1,
                method2_name: ci2
            },
            "effect_size_cohens_d": effect_size,
            "effect_size_interpretation": self._interpret_effect_size(effect_size)
        }
    
    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

