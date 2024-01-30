import torch
import numpy as np
import mobile_sam as msam
import coremltools as ct

def convert_image_encoder(model: msam.Sam, device: torch.device):
    input_tensor = torch.randn(1, 3, 1024, 1024).to(device)
    traced_image_encoder = torch.jit.trace(model.image_encoder, input_tensor)
    traced_image_encoder(input_tensor)
    
    coreml_model = ct.convert(
        traced_image_encoder,
        inputs=[ct.TensorType(name="image", shape=input_tensor.shape)],
        outputs=[ct.TensorType(name="imageEmbeddings")],
    )
    coreml_model.save("./coreml/ImageEncoder.mlpackage")

def convert_prompt_encoder(model: msam.Sam, device: torch.device):
    n = 2
    points = torch.randn(1, n, 2).to(device=device)
    labels = torch.ones(1, n).to(device=device)

    model.prompt_encoder.forward = model.prompt_encoder.coreml_forward

    traced_model = torch.jit.trace(model.prompt_encoder, (points, labels))
    traced_model(points, labels)

    r = ct.RangeDim(1, 100)

    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="points", shape=(1, r, 2)),
            ct.TensorType(name="labels", shape=(1, r))
        ],
        outputs=[
            ct.TensorType(name="sparsePromptEmbeddings"),
            ct.TensorType(name="densePromptEmbeddings")
        ]
    )
    coreml_model.save(f"./coreml/PromptEncoder.mlpackage")

def convert_mask_decoder(model: msam.Sam, device: torch.device):
    n = 3
    t1 = torch.randn(1, 256, 64, 64).to(device=device)
    t2 = torch.randn(1, 256, 64, 64).to(device=device)
    t3 = torch.randn(1, n, 256).to(device=device)
    t4 = torch.randn(1, 256, 64, 64).to(device=device)

    traced_mask_decoder = torch.jit.trace(model.mask_decoder, (t1, t2, t3, t4))
    traced_mask_decoder(t1, t2, t3, t4)

    coreml_model = ct.convert(
        traced_mask_decoder,
        inputs=[
            ct.TensorType(name="imageEmbeddings", shape=t1.shape),
            ct.TensorType(name="imagePositionalEncoding", shape=t2.shape),
            ct.TensorType(name="sparsePromptEmbeddings", shape=(1, ct.RangeDim(1, 100), 256)),
            ct.TensorType(name="densePromptEmbeddings", shape=t4.shape),
        ],
        outputs=[
            ct.TensorType(name="masks"),
            ct.TensorType(name="iouPredictions"),
        ],
    )
    coreml_model.save("./coreml/MaskDecoder.mlpackage")

def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS backend is not available.")
    
    device = torch.device("mps")

    model_type = "vit_t"
    model_checkpoint = "./weights/mobile_sam.pt"

    mobile_sam = msam.sam_model_registry[model_type](checkpoint=model_checkpoint)
    mobile_sam.to(device)
    mobile_sam.eval()

    convert_image_encoder(mobile_sam, device)
    convert_prompt_encoder(mobile_sam, device)
    convert_mask_decoder(mobile_sam, device)



if __name__ == '__main__':
    main()