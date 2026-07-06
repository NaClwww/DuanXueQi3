import argparse
from pathlib import Path

import torch

from hwdb_recognition.labels import LabelMap
from hwdb_recognition.model import HWDBCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/hwdb_cnn.onnx"), type=Path)
    args = parser.parse_args()

    labels = LabelMap.load(args.labels)
    model = HWDBCNN(num_classes=labels.size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


if __name__ == "__main__":
    main()
