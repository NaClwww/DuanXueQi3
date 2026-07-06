import cv2
import numpy as np


def normalize_character_image(image: np.ndarray, size: int = 64, padding: int = 6) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((1, size, size), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y : y + h, x : x + w]
    target = max(1, size - padding * 2)
    scale = min(target / max(w, 1), target / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return (canvas.astype(np.float32) / 255.0)[None, :, :]
