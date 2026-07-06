import tempfile
import unittest
from pathlib import Path

from hwdb_recognition.labels import LabelMap


class LabelMapTests(unittest.TestCase):
    def test_label_map_round_trips_json(self):
        label_map = LabelMap(["一", "丁", "七"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "labels.json"
            label_map.save(path)
            loaded = LabelMap.load(path)

        self.assertEqual(loaded.index_to_char(1), "丁")
        self.assertEqual(loaded.char_to_index("七"), 2)
        self.assertEqual(loaded.size, 3)


if __name__ == "__main__":
    unittest.main()
