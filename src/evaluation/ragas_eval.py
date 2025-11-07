from typing import List, Dict, Any
from dataclasses import dataclass
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import pandas as pd


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float


class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    async def evaluate_validation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> EvalResult:
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        
        result = evaluate(dataset, metrics=self.metrics)
        
        return EvalResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
            overall_score=sum([
                result["faithfulness"],
                result["answer_relevancy"],
                result["context_precision"],
                result["context_recall"]
            ]) / 4
        )
    
    def to_dataframe(self, results: List[EvalResult]) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in results])

