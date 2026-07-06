# HWDB Training Project

This directory contains the PyTorch training/export side for the on-device handwritten Chinese recognizer.

Expected production flow:

1. Download CASIA-HWDB from http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html.
2. Convert the offline `gnt` files into normalized `64x64` images and `labels.json`.
3. Train `HWDBCNN` with GB2312 level-1 labels, normally 3755 classes.
4. Export the checkpoint to ONNX with `hwdb_recognition/export_onnx.py`.
5. Convert ONNX to ncnn `hwdb_cnn.param` and `hwdb_cnn.bin`, then replace the demo files under `entry/src/main/resources/rawfile/`.

Install dependencies:

```powershell
python -m pip install -r training/requirements.txt
```

Run unit tests:

```powershell
$env:PYTHONPATH='training'
python -m unittest discover -s training\tests -t .
```

The current app includes a tiny ncnn demo model so the phone-side native inference path can be compiled and exercised before the CASIA-trained model is available.
