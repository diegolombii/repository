import numpy as np
import subprocess

onnx_file = "tcn_hif_autoencoder.onnx"
npz_file = "calibration_sani_ch6.npz"
output_folder = "tflite_output_int8"

# 1) Carico il file NPZ
data = np.load(npz_file)

# 2) Estraggo l'input di calibrazione
calib_data = data["input"].astype(np.float32)

print("Shape calibrazione:", calib_data.shape)
print("Tipo dati:", calib_data.dtype)

# 3) Salvo in formato NPY
calib_file = "calib_data.npy"
np.save(calib_file, calib_data)

# 4) Conversione ONNX -> TFLite INT8
subprocess.run([
    "onnx2tf",
    "-i", onnx_file,
    "-o", output_folder,
    "-oiqt",
    "-cind", "input", calib_file, "[0.0]", "[1.0]",
    "-tb", "tf_converter",
], check=True)

print("Conversione completata.")