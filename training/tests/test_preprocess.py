import unittest

import numpy as np

from hwdb_recognition.preprocess import normalize_character_image


class PreprocessTests(unittest.TestCase):
    def test_normalize_character_image_centers_foreground(self):
        image = np.full((80, 90), 255, dtype=np.uint8)
        image[20:50, 30:60] = 0

        output = normalize_character_image(image, size=64)

        self.assertEqual(output.shape, (1, 64, 64))
        self.assertEqual(output.dtype, np.float32)
        self.assertGreater(float(output[:, 20:44, 20:44].mean()), 0.2)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
