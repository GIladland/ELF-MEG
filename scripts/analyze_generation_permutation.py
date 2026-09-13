#!/usr/bin/env python
"""Permutation audit for one saved semantic-scaling generation result.

The null independently reassigns the generated sentences to target rows.  It
therefore preserves both text marginals while destroying the brain/text row
correspondence.  Scores are macro-averaged except corpus WER, whose reference
word denominator is fixed across permutations. BLEU-1 is macro sentence-level
modified unigram precision with the standard brevity penalty.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import site
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
ROUGE_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
ROUGE_VALID_TOKEN_RE = re.compile(r"^[a-z0-9]+$")
CONTENT_STOPWORDS = {
    "a", "about", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "quite", "really", "same", "she",
    "should", "so", "some", "such", "sure", "than", "that", "the", "their",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "we", "well", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "will", "with", "would", "yes", "you", "your", "yours",
    "yourself", "yourselves",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Semantic-scaling ladder JSON.")
    source.add_argument("--input-csv", help="CSV with target and generated columns.")
    source.add_argument(
        "--input-metrics-json",
        help="ELF best_metrics.json containing targets and generated arrays.",
    )
    parser.add_argument("--modality", default="meg_ada")
    parser.add_argument("--scale", default="x1")
    parser.add_argument("--rung", default="adapter")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--generated-top1", type=float, default=None)
    parser.add_argument("--generated-top5", type=float, default=None)
    parser.add_argument("--semantic-top1", type=float, default=None)
    parser.add_argument("--semantic-top5", type=float, default=None)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-model-path", default="")
    parser.add_argument("--bertscore-layers", type=int, default=17)
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--bertscore-batch-size", type=int, default=64)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [
        token
        for token in words(text)
        if token not in CONTENT_STOPWORDS and any(char.isalpha() for char in token)
    ]


def counted_f1(left: list[str], right: list[str]) -> tuple[float, list[str]]:
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = left_counts & right_counts
    count = sum(overlap.values())
    if not left or not right or count == 0:
        return 0.0, []
    precision = count / len(left)
    recall = count / len(right)
    expanded = [token for token, n in sorted(overlap.items()) for _ in range(n)]
    return 2 * precision * recall / (precision + recall), expanded


def sentence_bleu1(hypothesis: list[str], reference: list[str]) -> float:
    """Return single-reference BLEU-1 without an unnecessary smoothing choice."""
    if not hypothesis or not reference:
        return 0.0
    overlap = sum((Counter(hypothesis) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hypothesis)
    brevity_penalty = (
        1.0
        if len(hypothesis) > len(reference)
        else math.exp(1.0 - len(reference) / len(hypothesis))
    )
    return brevity_penalty * precision


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def porter_stemmer():
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer()
    except Exception:
        # Some analysis hosts have a usable pure-Python Porter module but a
        # binary-incompatible optional sklearn/pandas imported by nltk itself.
        candidates = []
        for root in [site.getusersitepackages(), *site.getsitepackages()]:
            candidates.append(Path(root) / "nltk" / "stem" / "porter.py")
        porter_path = next((path for path in candidates if path.exists()), None)
        if porter_path is None:
            raise RuntimeError("No Porter stemmer is available")
        for name in list(sys.modules):
            if name == "nltk" or name.startswith("nltk."):
                del sys.modules[name]
        nltk_module = types.ModuleType("nltk")
        stem_module = types.ModuleType("nltk.stem")
        api_module = types.ModuleType("nltk.stem.api")

        class StemmerI:
            pass

        api_module.StemmerI = StemmerI
        sys.modules.update(
            {"nltk": nltk_module, "nltk.stem": stem_module, "nltk.stem.api": api_module}
        )
        spec = importlib.util.spec_from_file_location("_standalone_porter", porter_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {porter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PorterStemmer()


def rouge_tokens(text: str, stemmer) -> list[str]:
    normalized = ROUGE_NON_ALNUM_RE.sub(" ", text.lower())
    tokens = normalized.split()
    tokens = [stemmer.stem(token) if len(token) > 3 else token for token in tokens]
    return [token for token in tokens if ROUGE_VALID_TOKEN_RE.match(token)]


def pairwise_matrices(generated: list[str], targets: list[str]) -> dict[str, np.ndarray]:
    n = len(targets)
    generated_words = [words(text) for text in generated]
    target_words = [words(text) for text in targets]
    generated_content = [content_words(text) for text in generated]
    target_content = [content_words(text) for text in targets]
    stemmer = porter_stemmer()
    generated_rouge = [rouge_tokens(text, stemmer) for text in generated]
    target_rouge = [rouge_tokens(text, stemmer) for text in targets]
    matrices = {
        "content_f1": np.zeros((n, n), dtype=np.float32),
        "word_f1": np.zeros((n, n), dtype=np.float32),
        "bleu1": np.zeros((n, n), dtype=np.float32),
        "rouge1_f1": np.zeros((n, n), dtype=np.float32),
        "wer_errors": np.zeros((n, n), dtype=np.float32),
    }
    for i in range(n):
        for j in range(n):
            matrices["content_f1"][i, j] = counted_f1(
                generated_content[i], target_content[j]
            )[0]
            matrices["word_f1"][i, j] = counted_f1(generated_words[i], target_words[j])[0]
            matrices["bleu1"][i, j] = sentence_bleu1(
                generated_words[i], target_words[j]
            )
            matrices["rouge1_f1"][i, j] = counted_f1(
                generated_rouge[i], target_rouge[j]
            )[0]
            matrices["wer_errors"][i, j] = edit_distance(target_words[j], generated_words[i])
    return matrices


def bertscore_matrix(generated: list[str], targets: list[str], args: argparse.Namespace) -> np.ndarray:
    from bert_score import BERTScorer

    model_type = args.bertscore_model_path or args.bertscore_model
    scorer = BERTScorer(
        model_type=model_type,
        num_layers=args.bertscore_layers,
        rescale_with_baseline=False,
        device=args.bertscore_device,
    )
    n = len(targets)
    candidates = [generated[i] for i in range(n) for _ in range(n)]
    references = [targets[j] for _ in range(n) for j in range(n)]
    _, _, f1 = scorer.score(
        candidates,
        references,
        batch_size=args.bertscore_batch_size,
        verbose=True,
    )
    return np.asarray(f1.tolist(), dtype=np.float32).reshape(n, n)


def binomial_tail(successes: int, trials: int, probability: float) -> float:
    return float(
        sum(
            math.comb(trials, k)
            * probability**k
            * (1.0 - probability) ** (trials - k)
            for k in range(successes, trials + 1)
        )
    )


def main() -> None:
    args = parse_args()
    if args.input_csv:
        with Path(args.input_csv).open(newline="", encoding="utf-8") as handle:
            input_rows = list(csv.DictReader(handle))
        result = {
            "run_tag": args.run_tag or Path(args.input_csv).stem,
            "checkpoint": args.checkpoint,
            "generated": [row["generated"] for row in input_rows],
            "targets": [row["target"] for row in input_rows],
            "generated_top1": args.generated_top1,
            "generated_top5": args.generated_top5,
            "semantic_top1": args.semantic_top1,
            "semantic_top5": args.semantic_top5,
        }
    elif args.input_metrics_json:
        payload = json.loads(Path(args.input_metrics_json).read_text(encoding="utf-8"))
        generated_retrieval = payload.get("generation_t5_retrieval", {})
        result = {
            "run_tag": args.run_tag or Path(args.input_metrics_json).parent.name,
            "checkpoint": args.checkpoint,
            "generated": payload["generated"],
            "targets": payload["targets"],
            "generated_top1": (
                args.generated_top1
                if args.generated_top1 is not None
                else generated_retrieval.get("top1")
            ),
            "generated_top5": (
                args.generated_top5
                if args.generated_top5 is not None
                else generated_retrieval.get("top5")
            ),
            # Interface-monitor retrieval can be computed on a small diagnostic
            # batch rather than all rows, so never infer a full-set rate from it.
            "semantic_top1": args.semantic_top1,
            "semantic_top5": args.semantic_top5,
        }
    else:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = next(
            row
            for row in payload["modalities"][args.modality]
            if row.get("scale") == args.scale and row.get("rung") == args.rung
        )
    generated = list(map(str, result["generated"]))
    targets = list(map(str, result["targets"]))
    if len(generated) != len(targets):
        raise ValueError("generated/targets length mismatch")
    n = len(targets)
    matrices = pairwise_matrices(generated, targets)
    # Preserve the same row permutation for both terms of the predeclared
    # lexical selection metric.
    matrices["word_content_f1_sum"] = (
        matrices["word_f1"] + matrices["content_f1"]
    )
    if args.bertscore:
        matrices["bertscore_raw_f1"] = bertscore_matrix(generated, targets, args)

    rng = np.random.default_rng(args.seed)
    row_indices = np.arange(n)
    null = {name: np.empty(args.permutations, dtype=np.float32) for name in matrices}
    reference_words = sum(len(words(text)) for text in targets)
    for index in range(args.permutations):
        permutation = rng.permutation(n)
        for name, matrix in matrices.items():
            value = float(matrix[row_indices, permutation].sum())
            null[name][index] = value / (reference_words if name == "wer_errors" else n)

    tests = {}
    for name, matrix in matrices.items():
        observed = float(np.trace(matrix)) / (reference_words if name == "wer_errors" else n)
        values = null[name]
        lower_is_better = name == "wer_errors"
        extreme = values <= observed if lower_is_better else values >= observed
        tests["wer" if lower_is_better else name] = {
            "observed": observed,
            "null_mean": float(values.mean()),
            "null_std": float(values.std(ddof=1)),
            "null_ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "one_sided_p": float((np.count_nonzero(extreme) + 1) / (len(values) + 1)),
            "direction": "lower" if lower_is_better else "higher",
        }

    retrieval = {
        "note": "Exact one-sided binomial chance tests; these are not row-permutation tests.",
    }
    for prefix in ("generated", "semantic"):
        for k in (1, 5):
            key = f"{prefix}_top{k}"
            rate = result.get(key)
            if rate is None:
                continue
            hits = round(float(rate) * n)
            chance = k / n
            retrieval[key] = {
                "hits": hits,
                "rate": float(rate),
                "chance": chance,
                "p": binomial_tail(hits, n, chance),
            }

    rows = []
    for index, (target, prediction) in enumerate(zip(targets, generated)):
        content_f1, overlap = counted_f1(content_words(prediction), content_words(target))
        word_f1, _ = counted_f1(words(prediction), words(target))
        bleu1 = float(matrices["bleu1"][index, index])
        rouge1_f1 = float(matrices["rouge1_f1"][index, index])
        row = {
            "index": index,
            "target": target,
            "generated": prediction,
            "content_f1": content_f1,
            "word_f1": word_f1,
            "bleu1": bleu1,
            "rouge1_f1": rouge1_f1,
            "overlapping_content_words": ", ".join(overlap),
        }
        if "bertscore_raw_f1" in matrices:
            row["bertscore_raw_f1"] = float(matrices["bertscore_raw_f1"][index, index])
        rows.append(row)
    rows.sort(key=lambda row: (row["content_f1"], row["word_f1"]), reverse=True)

    output = {
        "contract": {
            "run_tag": result["run_tag"],
            "checkpoint": result["checkpoint"],
            "n": n,
            "null": "random one-to-one reassignment of generated sentences to target rows",
            "permutations": args.permutations,
            "seed": args.seed,
            "bleu1": "macro sentence BLEU-1 with modified unigram precision and brevity penalty",
            "rouge1": "macro ROUGE-1 F1 with Porter stemming",
            "bertscore": (
                f"raw non-rescaled {args.bertscore_model}, layer {args.bertscore_layers}"
                if args.bertscore
                else "not computed"
            ),
        },
        "permutation_tests": tests,
        "retrieval_chance_tests": retrieval,
        "top_generations": rows[:10],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = Path(args.output_csv)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
