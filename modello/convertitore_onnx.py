import os

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


def remove_initializers_from_graph_inputs(onnx_path: str):
    """
    Rimuove dagli input del grafo eventuali pesi/initializer esportati anche
    come graph input. Serve a evitare i warning di ONNX Runtime del tipo:
    'Initializer ... appears in graph inputs'.
    """
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "Installa il pacchetto onnx per eseguire la pulizia: pip install onnx"
        ) from exc

    model = onnx.load(onnx_path)
    initializer_names = {initializer.name for initializer in model.graph.initializer}

    real_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]

    removed = len(model.graph.input) - len(real_inputs)
    del model.graph.input[:]
    model.graph.input.extend(real_inputs)

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)

    print(f"Initializer rimossi dai graph input: {removed}")
    print("Input reali del modello:", [x.name for x in model.graph.input])


def main():
    pt_path = "best_tcn_ae.pt"
    onnx_path = "tcn_hif_autoencoder.onnx"
    frame_len = 600
    n_features = 6
    opset = 12

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"File .pt non trovato: {pt_path}")

    model = build_model(n_features)
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Input ONNX fisso: (batch=1, lunghezza=600, feature=6)
    dummy_input = torch.randn(1, frame_len, n_features, dtype=torch.float32)

    with torch.no_grad():
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

    # Pulizia esplicita per evitare initializer trattati come input dal runtime.
    remove_initializers_from_graph_inputs(onnx_path)

    print(f"Modello esportato in ONNX: {onnx_path}")
    print("Input atteso: input -> (1, 600, 6), float32")
    print("Output atteso: reconstruction -> (1, 600, 6)")


if __name__ == "__main__":
    main()