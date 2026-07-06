import unittest

import torch

from hwdb_recognition.model import HWDBCNN


class ModelTests(unittest.TestCase):
    def test_hwdb_cnn_outputs_one_logit_per_class(self):
        model = HWDBCNN(num_classes=3755)
        model.eval()

        with torch.no_grad():
            logits = model(torch.zeros(2, 1, 64, 64))

        self.assertEqual(tuple(logits.shape), (2, 3755))


if __name__ == "__main__":
    unittest.main()
