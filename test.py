"""Run Marabou robustness checks for the exported Wine MLP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from maraboupy import Marabou, MarabouCore, MarabouUtils
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/wine_mlp.onnx"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/wine_mlp_metadata.json"),
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        help="Perturbation radii in the normalized feature space.",
    )
    parser.add_argument(
        "--sample-source",
        choices=["low-margin", "stored"],
        default="low-margin",
        help="Use low-margin correct test samples or the sample saved in metadata.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of low-margin correct test samples to verify.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/wine_marabou_results.json"),
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python src/train_wine_mlp.py` before test.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_flat_vars(var_array: Any) -> list[int]:
    return [int(var) for var in np.array(var_array).flatten()]


def margin_for_logits(logits: np.ndarray, predicted_class: int) -> float:
    other_logits = np.delete(logits, predicted_class)
    return float(logits[predicted_class] - np.max(other_logits))


def reconstruct_test_set(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_wine()
    x = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)
    indices = np.arange(len(y))

    _x_train, x_test, _y_train, y_test, _train_indices, test_indices = train_test_split(
        x,
        y,
        indices,
        test_size=0.25,
        random_state=int(metadata["seed"]),
        stratify=y,
    )

    scaler_mean = np.array(metadata["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(metadata["scaler_scale"], dtype=np.float32)
    x_test_scaled = ((x_test - scaler_mean) / scaler_scale).astype(np.float32)
    return x_test, x_test_scaled, y_test, test_indices


def run_onnx(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_array = inputs.astype(np.float32)
    outputs = [
        session.run(None, {input_name: input_array[index : index + 1]})[0][0]
        for index in range(input_array.shape[0])
    ]
    return np.array(outputs, dtype=np.float32)


def low_margin_samples(
    model_path: Path,
    metadata: dict[str, Any],
    sample_count: int,
) -> list[dict[str, Any]]:
    x_test, x_test_scaled, y_test, test_indices = reconstruct_test_set(metadata)
    logits = run_onnx(model_path, x_test_scaled)
    predictions = np.argmax(logits, axis=1)
    correct_indices = np.where(predictions == y_test)[0]
    if correct_indices.size == 0:
        raise RuntimeError("No correctly classified test samples were found.")

    candidates = []
    for test_position in correct_indices:
        predicted_class = int(predictions[test_position])
        candidates.append(
            {
                "test_index": int(test_position),
                "dataset_index": int(test_indices[test_position]),
                "true_class": int(y_test[test_position]),
                "predicted_class": predicted_class,
                "logits": logits[test_position].astype(float).tolist(),
                "margin": margin_for_logits(logits[test_position], predicted_class),
                "original_input": x_test[test_position].astype(float).tolist(),
                "normalized_input": x_test_scaled[test_position].astype(float).tolist(),
            }
        )

    candidates.sort(key=lambda sample: sample["margin"])
    selected = candidates[:sample_count]
    for index, sample in enumerate(selected, start=1):
        sample["sample_id"] = f"low_margin_{index}"
        sample["rank_by_margin"] = index
    return selected


def stored_sample(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    sample = dict(metadata["verification_sample"])
    sample["sample_id"] = "metadata_reference"
    return [sample]


def add_target_beats_original_constraint(
    network: Any,
    output_vars: list[int],
    original_class: int,
    target_class: int,
) -> None:
    # Search for a counterexample where y_target >= y_original.
    equation = MarabouUtils.Equation(MarabouCore.Equation.LE)
    equation.addAddend(1.0, output_vars[original_class])
    equation.addAddend(-1.0, output_vars[target_class])
    equation.setScalar(0.0)
    network.addEquation(equation)


def solve_single_target(
    model_path: Path,
    sample: list[float],
    epsilon: float,
    original_class: int,
    target_class: int,
    timeout: int,
) -> dict[str, Any]:
    network = Marabou.read_onnx(str(model_path))
    input_vars = get_flat_vars(network.inputVars[0])
    output_vars = get_flat_vars(network.outputVars[0])

    if len(input_vars) != len(sample):
        raise RuntimeError(
            f"Expected {len(sample)} input variables, got {len(input_vars)}."
        )
    if original_class >= len(output_vars) or target_class >= len(output_vars):
        raise RuntimeError("Class index is outside the ONNX output shape.")

    for variable, value in zip(input_vars, sample):
        network.setLowerBound(variable, float(value - epsilon))
        network.setUpperBound(variable, float(value + epsilon))

    add_target_beats_original_constraint(
        network=network,
        output_vars=output_vars,
        original_class=original_class,
        target_class=target_class,
    )

    options = Marabou.createOptions(timeoutInSeconds=timeout, verbosity=0)
    start = time.perf_counter()
    exit_code, values, _stats = network.solve(options=options, verbose=False)
    elapsed = time.perf_counter() - start

    sat = exit_code == "sat"
    result: dict[str, Any] = {
        "epsilon": epsilon,
        "target_class": target_class,
        "status": str(exit_code).upper(),
        "runtime_seconds": elapsed,
    }

    if sat:
        result["counterexample_input"] = [float(values[var]) for var in input_vars]
        result["counterexample_outputs"] = [float(values[var]) for var in output_vars]

    return result


def add_original_counterexample(
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    if "counterexample_input" not in result:
        return

    scaler_mean = np.array(metadata["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(metadata["scaler_scale"], dtype=np.float32)
    normalized = np.array(result["counterexample_input"], dtype=np.float32)
    original = normalized * scaler_scale + scaler_mean
    result["counterexample_original_input"] = original.astype(float).tolist()


def verify_sample(
    model_path: Path,
    sample: dict[str, Any],
    class_names: list[str],
    epsilons: list[float],
    timeout: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    original_class = int(sample["predicted_class"])
    epsilon_results = []

    for epsilon in epsilons:
        target_results = []
        for target_class in range(len(class_names)):
            if target_class == original_class:
                continue
            result = solve_single_target(
                model_path=model_path,
                sample=sample["normalized_input"],
                epsilon=epsilon,
                original_class=original_class,
                target_class=target_class,
                timeout=timeout,
            )
            add_original_counterexample(result, metadata)
            target_results.append(result)

        epsilon_results.append(
            {
                "epsilon": epsilon,
                "robust": all(result["status"] == "UNSAT" for result in target_results),
                "target_results": target_results,
            }
        )

    robust_epsilons = [
        result["epsilon"] for result in epsilon_results if result["robust"]
    ]
    non_robust_epsilons = [
        result["epsilon"] for result in epsilon_results if not result["robust"]
    ]

    return {
        "sample": sample,
        "summary": {
            "largest_tested_robust_epsilon": max(robust_epsilons)
            if robust_epsilons
            else None,
            "first_non_robust_epsilon": min(non_robust_epsilons)
            if non_robust_epsilons
            else None,
        },
        "epsilon_results": epsilon_results,
    }


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    metadata_path = args.metadata.resolve()

    if not model_path.exists():
        raise SystemExit(
            f"Missing {model_path}. Run `python src/train_wine_mlp.py` first."
        )

    metadata = load_metadata(metadata_path)
    class_names = metadata["target_names"]
    if args.sample_source == "low-margin":
        samples = low_margin_samples(model_path, metadata, args.sample_count)
    else:
        samples = stored_sample(metadata)

    print("Wine MLP Marabou verification")
    print(f"Model: {model_path}")
    print(f"Sample source: {args.sample_source}")
    print(f"Testing epsilons: {args.epsilons}")
    print()
    print("Selected samples:")
    for sample in samples:
        predicted = int(sample["predicted_class"])
        print(
            f"- {sample['sample_id']}: test_index={sample.get('test_index')}, "
            f"predicted={predicted} ({class_names[predicted]}), "
            f"margin={sample['margin']:.6f}"
        )
    print()

    verification_results = []
    for sample in samples:
        predicted = int(sample["predicted_class"])
        print(
            f"Verifying {sample['sample_id']} "
            f"(predicted={predicted}, margin={sample['margin']:.6f})"
        )
        sample_result = verify_sample(
            model_path=model_path,
            sample=sample,
            class_names=class_names,
            epsilons=args.epsilons,
            timeout=args.timeout,
            metadata=metadata,
        )
        verification_results.append(sample_result)

        for epsilon_result in sample_result["epsilon_results"]:
            epsilon = epsilon_result["epsilon"]
            for result in epsilon_result["target_results"]:
                target_class = result["target_class"]
                print(
                    f"  epsilon={epsilon:.4f}, target={target_class} "
                    f"({class_names[target_class]}): {result['status']} "
                    f"in {result['runtime_seconds']:.3f}s"
                )
            print(f"  epsilon={epsilon:.4f} robust: {epsilon_result['robust']}")
        print()

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "metadata": str(args.metadata),
                "sample_source": args.sample_source,
                "sample_count": len(samples),
                "epsilons": args.epsilons,
                "class_names": class_names,
                "samples": verification_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
