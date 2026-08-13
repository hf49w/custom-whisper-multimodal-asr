"""Train a validation-set candidate reranker for A9 n-best hypotheses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    DEFAULT_FEATURE_NAMES,
    ClipCandidateScorer,
    ensure_candidate_scores,
    flatten_candidate_table,
    oracle_curve,
    prediction_metrics,
    read_jsonl,
    resolve_cross_platform_path,
    save_pickle,
    select_model_predictions,
    write_json,
    write_jsonl,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--clip-model-name", default="")
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--model-type", choices=["ridge", "gbr", "auto"], default="auto")
    parser.add_argument("--ridge-alpha-values", default="0.01,0.1,1,10,100")
    parser.add_argument("--gbr-n-estimators", default="50,100")
    parser.add_argument("--gbr-max-depth", default="2,3")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--feature-names", default="")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-scored-jsonl", action="store_true")
    return parser.parse_args()


def comma_floats(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def comma_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_feature_names(text: str) -> List[str]:
    if not text:
        return list(DEFAULT_FEATURE_NAMES)
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("--feature-names produced an empty list")
    return names


def build_estimator(model_type: str, params: Dict[str, Any]):
    if model_type == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(params["alpha"]))
    if model_type == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=0,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_predict_fold(
    *,
    model_type: str,
    params: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
) -> Tuple[np.ndarray, Any, Any]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    if model_type == "ridge":
        x_train_used = scaler.fit_transform(x_train)
        x_valid_used = scaler.transform(x_valid)
    else:
        scaler.fit(x_train)
        x_train_used = x_train
        x_valid_used = x_valid
    model = build_estimator(model_type, params)
    model.fit(x_train_used, y_train)
    return model.predict(x_valid_used), model, scaler


def grouped_folds(groups: np.ndarray, requested_folds: int):
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(max(2, int(requested_folds)), len(unique_groups))
    return GroupKFold(n_splits=n_splits).split(np.zeros_like(groups), groups=groups)


def candidate_configs(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    configs: List[Tuple[str, Dict[str, Any]]] = []
    if args.model_type in {"ridge", "auto"}:
        for alpha in comma_floats(args.ridge_alpha_values):
            configs.append(("ridge", {"alpha": alpha}))
    if args.model_type in {"gbr", "auto"}:
        for n_estimators in comma_ints(args.gbr_n_estimators):
            for max_depth in comma_ints(args.gbr_max_depth):
                configs.append(
                    (
                        "gbr",
                        {"n_estimators": n_estimators, "max_depth": max_depth},
                    )
                )
    if not configs:
        raise ValueError("No model configs generated")
    return configs


def evaluate_config(
    samples: Sequence[Dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    index_pairs: Sequence[Tuple[int, int]],
    *,
    model_type: str,
    params: Dict[str, Any],
    cv_folds: int,
) -> Dict[str, Any]:
    all_scores = np.zeros(x.shape[0], dtype=np.float32)
    fold_metrics: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(grouped_folds(groups, cv_folds), start=1):
        scores, _model, _scaler = fit_predict_fold(
            model_type=model_type,
            params=params,
            x_train=x[train_idx],
            y_train=y[train_idx],
            x_valid=x[valid_idx],
        )
        all_scores[valid_idx] = scores.astype(np.float32)
        valid_pairs = [index_pairs[int(i)] for i in valid_idx]
        valid_sample_indices = sorted({pair[0] for pair in valid_pairs})
        valid_samples = [samples[i] for i in valid_sample_indices]

        # Re-map valid scores/index pairs to the compact valid sample list.
        remap = {old: new for new, old in enumerate(valid_sample_indices)}
        compact_pairs = [(remap[pair[0]], pair[1]) for pair in valid_pairs]
        predictions = select_model_predictions(
            valid_samples,
            np.asarray(scores, dtype=np.float32),
            compact_pairs,
            selector=f"cv_fold_{fold_idx}",
        )
        fold_metrics.append({"fold": fold_idx, **prediction_metrics(predictions)})

    predictions = select_model_predictions(
        samples,
        all_scores,
        index_pairs,
        selector=f"cv_{model_type}",
    )
    metrics = prediction_metrics(predictions)
    return {
        "model_type": model_type,
        "params": dict(params),
        **metrics,
        "fold_metrics": fold_metrics,
    }


def fit_final_model(
    *,
    model_type: str,
    params: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[Any, Any]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    if model_type == "ridge":
        x_used = scaler.fit_transform(x)
    else:
        scaler.fit(x)
        x_used = x
    model = build_estimator(model_type, params)
    model.fit(x_used, y)
    return model, scaler


def main() -> None:
    args = parse_args()
    val_jsonl = resolve_cross_platform_path(args.val_jsonl)
    output_pkl = resolve_cross_platform_path(args.output_pkl)
    output_dir = (
        resolve_cross_platform_path(args.output_dir)
        if args.output_dir
        else output_pkl.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(val_jsonl)
    if not samples:
        raise ValueError(f"No samples loaded from {val_jsonl}")

    clip_scorer = None
    if args.clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(
            str(resolve_cross_platform_path(args.clip_model_name)),
            device=device,
            no_download=args.no_download,
        )
    samples = ensure_candidate_scores(samples, clip_scorer=clip_scorer, log_every=args.log_every)
    if args.save_scored_jsonl:
        write_jsonl(output_dir / "val_scored_candidates.jsonl", samples)

    feature_names = parse_feature_names(args.feature_names)
    x, y, groups, index_pairs, _sample_ids = flatten_candidate_table(
        samples,
        feature_names=feature_names,
        with_labels=True,
    )

    cv_records: List[Dict[str, Any]] = []
    for model_type, params in candidate_configs(args):
        record = evaluate_config(
            samples,
            x,
            y,
            groups,
            index_pairs,
            model_type=model_type,
            params=params,
            cv_folds=args.cv_folds,
        )
        cv_records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    best = min(cv_records, key=lambda record: (record["wer"], record["cer"]))
    model, scaler = fit_final_model(
        model_type=best["model_type"],
        params=best["params"],
        x=x,
        y=y,
    )
    x_used = scaler.transform(x) if best["model_type"] == "ridge" else x
    train_scores = model.predict(x_used)
    train_predictions = select_model_predictions(
        samples,
        np.asarray(train_scores, dtype=np.float32),
        index_pairs,
        selector="train_fit",
    )

    payload = {
        "kind": "a9_candidate_reranker",
        "model_type": best["model_type"],
        "params": best["params"],
        "feature_names": feature_names,
        "model": model,
        "scaler": scaler,
        "clip_model_name": str(resolve_cross_platform_path(args.clip_model_name)) if args.clip_model_name else "",
        "no_download": bool(args.no_download),
        "cv_best": best,
        "cv_records": cv_records,
    }
    save_pickle(output_pkl, payload)

    write_predictions(output_dir / "val_predictions_train_fit.jsonl", train_predictions)
    summary = {
        "val_jsonl": str(val_jsonl),
        "output_pkl": str(output_pkl),
        "rows": len(samples),
        "candidate_rows": int(x.shape[0]),
        "feature_names": feature_names,
        "top1": prediction_metrics(
            [select_model_predictions([sample], np.asarray([0.0] * len(sample.get("candidates", []))), [(0, idx) for idx in range(len(sample.get("candidates", [])))], selector="top1")[0] for sample in samples if sample.get("candidates")]
        ),
        "oracle_curve": oracle_curve(samples, [1, 5, 10, 20, 30, 50]),
        "cv_best": best,
        "train_fit": prediction_metrics(train_predictions),
    }
    write_json(output_dir / "train_summary.json", summary)
    write_jsonl(output_dir / "cv_metrics.jsonl", cv_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
