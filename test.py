"""Run Marabou robustness checks for the exported Wine MLP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from maraboupy import Marabou, MarabouCore


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
        default=[0.01, 0.03, 0.05],
        help="Perturbation radii in the normalized feature space.",
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


def add_target_beats_original_constraint(
    network: Any,
    output_vars: list[int],
    original_class: int,
    target_class: int,
) -> None:
    # Search for a counterexample where y_target >= y_original.
    equation = MarabouCore.Equation(MarabouCore.Equation.LE)
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

    options = Marabou.createOptions(timeoutInSeconds=timeout)
    start = time.perf_counter()
    values, _stats = network.solve(options=options, verbose=False)
    elapsed = time.perf_counter() - start

    sat = len(values) > 0
    result: dict[str, Any] = {
        "epsilon": epsilon,
        "target_class": target_class,
        "status": "SAT" if sat else "UNSAT",
        "runtime_seconds": elapsed,
    }

    if sat:
        result["counterexample_input"] = [float(values[var]) for var in input_vars]
        result["counterexample_outputs"] = [float(values[var]) for var in output_vars]

    return result


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    metadata_path = args.metadata.resolve()

    if not model_path.exists():
        raise SystemExit(
            f"Missing {model_path}. Run `python src/train_wine_mlp.py` first."
        )

    metadata = load_metadata(metadata_path)
    sample = metadata["verification_sample"]["normalized_input"]
    original_class = int(metadata["verification_sample"]["predicted_class"])
    class_names = metadata["target_names"]

    print("Wine MLP Marabou verification")
    print(f"Model: {model_path}")
    print(f"Original class: {original_class} ({class_names[original_class]})")
    print(f"Testing epsilons: {args.epsilons}")
    print()

    all_results = []
    for epsilon in args.epsilons:
        epsilon_results = []
        for target_class in range(len(class_names)):
            if target_class == original_class:
                continue
            result = solve_single_target(
                model_path=model_path,
                sample=sample,
                epsilon=epsilon,
                original_class=original_class,
                target_class=target_class,
                timeout=args.timeout,
            )
            epsilon_results.append(result)
            print(
                f"epsilon={epsilon:.4f}, target={target_class} "
                f"({class_names[target_class]}): {result['status']} "
                f"in {result['runtime_seconds']:.3f}s"
            )

        robust = all(result["status"] == "UNSAT" for result in epsilon_results)
        print(f"epsilon={epsilon:.4f} robust: {robust}")
        print()
        all_results.extend(epsilon_results)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "metadata": str(args.metadata),
                "original_class": original_class,
                "class_names": class_names,
                "results": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
