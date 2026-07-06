from pathlib import Path
import math
import struct


ROOT = Path(__file__).resolve().parents[1]
RAWFILE = ROOT / "entry" / "src" / "main" / "resources" / "rawfile"
CLASSES = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9"]
INPUT_SIZE = 64 * 64


def write_param(path: Path) -> None:
    text = f"""7767517
4 4
Input            input            0 1 input 0=1 1=64 2=64
Flatten          flatten          1 1 input flat
InnerProduct     fc               1 1 flat logits 0={len(CLASSES)} 1=1 2={len(CLASSES) * INPUT_SIZE}
Softmax          prob             1 1 logits prob 0=0
"""
    path.write_text(text, encoding="utf-8")


def write_bin(path: Path) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<I", 0))
        for cls_index in range(len(CLASSES)):
            for pixel_index in range(INPUT_SIZE):
                row = pixel_index // 64
                col = pixel_index % 64
                angle = (cls_index + 1) * 0.021
                value = math.sin(row * angle) * 0.002 + math.cos(col * angle) * 0.002
                file.write(struct.pack("<f", value))

        file.write(struct.pack("<I", 0))
        for cls_index in range(len(CLASSES)):
            file.write(struct.pack("<f", 0.01 * cls_index))


def write_labels(path: Path) -> None:
    path.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def main() -> None:
    RAWFILE.mkdir(parents=True, exist_ok=True)
    write_param(RAWFILE / "hwdb_demo.param")
    write_bin(RAWFILE / "hwdb_demo.bin")
    write_labels(RAWFILE / "hwdb_demo_labels.txt")


if __name__ == "__main__":
    main()
