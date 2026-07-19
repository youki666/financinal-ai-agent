"""RAGAS 评估 CLI 入口

用法:
    python run_eval.py                          # 使用内置数据集，全部三个指标
    python run_eval.py --dataset custom.json    # 使用自定义数据集
    python run_eval.py --metrics faithfulness   # 仅评估 Faithfulness
    python run_eval.py --language chinese       # 使用中文 prompt 模板
    python run_eval.py --output results.json    # 将结果保存为 JSON
    python run_eval.py --verbose                # 打印每题详情
"""

import argparse
import json
import sys

from evaluation.eval_runner import run_evaluation, AVAILABLE_METRICS


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估工具")
    parser.add_argument(
        "--dataset", type=str, default=None, help="评估数据集 JSON 文件路径"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="faithfulness,answer_relevancy,context_recall",
        help="逗号分隔的指标名称",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="结果输出 JSON 文件路径"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="打印每题详细得分"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "chinese"],
        help="prompt 模板语言（默认 english）",
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    invalid = [m for m in metrics if m not in AVAILABLE_METRICS]
    if invalid:
        print(f"未知指标: {invalid}. 可选: {list(AVAILABLE_METRICS.keys())}")
        sys.exit(1)

    result = run_evaluation(dataset_path=args.dataset, metrics=metrics, language=args.language)

    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    for metric, score in result.metric_scores.items():
        print(f"  {metric}: {score:.4f}")

    if args.verbose:
        print("\n" + "-" * 60)
        print("每题详情")
        print("-" * 60)
        for entry in result.per_question:
            print(f"\n  Q: {entry['question']}")
            for m in metrics:
                val = entry.get(m)
                if val is not None:
                    print(f"    {m}: {val:.4f}")

    if args.output:
        output_data = {
            "summary": result.metric_scores,
            "details": result.per_question,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
