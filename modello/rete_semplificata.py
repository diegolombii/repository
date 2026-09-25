import torch
from tcn_rel_earlystop_patch5_perunit import TCNAutoencoder


def build_model(n_features: int) -> TCNAutoencoder:
    return TCNAutoencoder(
        n_features=n_features,
        hidden_ch=56,
        latent_ch=16,
        kernel_size=5,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
        use_downsample=True,
    )


def main():
    onnx_path = "tcn_hif_autoencoder_semplificato.onnx"

    frame_len = 600
    n_features = 6
    opset = 12

    # Creo il modello nuovo con pesi randomici
    model = build_model(n_features)
    model.eval()

    # Input ONNX: batch=1, lunghezza=600, feature=6
    dummy_input = torch.randn(1, frame_len, n_features, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["reconstruction"],
        opset_version=opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )

if __name__ == "__main__":
    main()