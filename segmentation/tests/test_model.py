import torch
from model import TinyUNet


def test_forward_pass_output_shape():
    model = TinyUNet(num_classes=3, base_width=4)  # tiny for a fast test
    x = torch.rand(2, 3, 64, 64)

    logits = model(x)

    assert logits.shape == (2, 3, 64, 64)


def test_forward_pass_is_deterministic_in_eval_mode():
    model = TinyUNet(num_classes=3, base_width=4)
    model.eval()
    x = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.equal(out1, out2)
