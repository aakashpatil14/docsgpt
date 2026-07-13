"""RAGAS evaluation of the full RAG pipeline (the RAG Triad + context recall).

Runs each golden question through the live pipeline, then scores with RAGAS:
  - Faithfulness            (is the answer grounded in retrieved context?)
  - ResponseRelevancy       (does the answer address the question?)
  - ContextPrecision        (are the retrieved chunks on-topic?)
  - ContextRecall           (was the needed info retrieved, vs the reference?)

Requires the OpenAI backend (judge LLM + embeddings) and an ingested corpus:

    pip install ragas datasets langchain-openai
    python -m experiments.ragas_eval
"""
import os

import config
from experiments.eval_dataset import EVAL_DATASET
from generation.answer import answer_question

RESULTS_CSV = os.path.join("data", "ragas_results.csv")


def _load_ragas():
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Missing eval deps. Run: pip install ragas datasets langchain-openai\n"
            f"({e})"
        )
    return locals()


def main():
    r = _load_ragas()

    print(f"Running pipeline over {len(EVAL_DATASET)} questions...")
    samples = []
    for item in EVAL_DATASET:
        res = answer_question(item["question"])
        samples.append(
            r["SingleTurnSample"](
                user_input=item["question"],
                response=res["answer"],
                retrieved_contexts=res["contexts"],
                reference=item["reference"],
            )
        )

    dataset = r["EvaluationDataset"](samples=samples)
    judge_llm = r["LangchainLLMWrapper"](r["ChatOpenAI"](model=config.LLM_MODEL, temperature=0))
    judge_emb = r["LangchainEmbeddingsWrapper"](
        r["OpenAIEmbeddings"](model="text-embedding-3-small")
    )

    metrics = [
        r["Faithfulness"](),
        r["ResponseRelevancy"](),
        r["LLMContextPrecisionWithReference"](),
        r["LLMContextRecall"](),
    ]

    print("Scoring with RAGAS (this makes LLM judge calls)...")
    result = r["evaluate"](
        dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_emb
    )
    print("\n=== RAGAS scores ===")
    print(result)

    os.makedirs("data", exist_ok=True)
    result.to_pandas().to_csv(RESULTS_CSV, index=False)
    print(f"\nPer-question results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
