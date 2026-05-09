"""Train a small Wine MLP and export it to ONNX.

The exported model intentionally ends with logits, not softmax probabilities.
This keeps Marabou output constraints linear: for example, y_j >= y_c.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


class WineMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_verification_sample(
    logits: np.ndarray,
    labels: np.ndarray,
    original_inputs: np.ndarray,
    normalized_inputs: np.ndarray,
) -> dict:
    predictions = np.argmax(logits, axis=1)
    correct_indices = np.where(predictions == labels)[0]
    if correct_indices.size == 0:
        raise RuntimeError("No correctly classified test sample was found.")

    margins = []
    for index in correct_indices:
        predicted = int(predictions[index])
        other_logits = np.delete(logits[index], predicted)
        margins.append(float(logits[index, predicted] - np.max(other_logits)))

    selected_position = int(correct_indices[int(np.argmax(margins))])
    return {
        "test_index": selected_position,
        "true_class": int(labels[selected_position]),
        "predicted_class": int(predictions[selected_position]),
        "logits": logits[selected_position].astype(float).tolist(),
        "margin": float(max(margins)),
        "original_input": original_inputs[selected_position].astype(float).tolist(),
        "normalized_input": normalized_inputs[selected_position].astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    dataset = load_wine()
    x = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    model = WineMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_inputs = torch.from_numpy(x_train_scaled)
    train_labels = torch.from_numpy(y_train)

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_inputs), train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(torch.from_numpy(x_train_scaled)).numpy()
        test_logits = model(torch.from_numpy(x_test_scaled)).numpy()

    train_predictions = np.argmax(train_logits, axis=1)
    test_predictions = np.argmax(test_logits, axis=1)
    train_accuracy = float(accuracy_score(y_train, train_predictions))
    test_accuracy = float(accuracy_score(y_test, test_predictions))

    onnx_path = output_dir / "wine_mlp.onnx"
    metadata_path = output_dir / "wine_mlp_metadata.json"
    dummy_input = torch.zeros((1, 13), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": x_test_scaled[:1]})[0]
    pytorch_logits = test_logits[:1]
    max_abs_diff = float(np.max(np.abs(onnx_logits - pytorch_logits)))

    selected = select_verification_sample(
        logits=test_logits,
        labels=y_test,
        original_inputs=x_test,
        normalized_inputs=x_test_scaled,
    )

    metadata = {
        "dataset": "sklearn.datasets.load_wine",
        "architecture": "13 -> 16 -> ReLU -> 8 -> ReLU -> 3",
        "onnx_model": str(onnx_path.relative_to(project_root)),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "onnx_max_abs_diff": max_abs_diff,
        "default_epsilon": args.epsilon,
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "verification_sample": selected,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved ONNX model to {onnx_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"ONNX/PyTorch max abs diff: {max_abs_diff:.8f}")
    print(
        "Selected verification class: "
        f"{selected['predicted_class']} ({dataset.target_names[selected['predicted_class']]})"
    )


if __name__ == "__main__":
    main()
