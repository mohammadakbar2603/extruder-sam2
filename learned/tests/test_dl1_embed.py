"""Tests for learned_pathway.dl1_embed (DL-1): DINOv2 input preparation and embedding.

No test downloads anything: embed() runs against a stub model, and load_dinov2 is not called.
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from learned_pathway import dl1_embed  # noqa: E402


def test_t13_to_model_input_normalises_like_imagenet():
    """T13: /255 then per-channel (value - mean) / std; grey is copied to all three channels."""
    grey = np.full((2, 224, 224), 255, np.uint8)
    x = dl1_embed.to_model_input(grey)
    assert x.shape == (2, 3, 224, 224)
    assert x.dtype == np.float32
    for c in range(3):
        assert np.allclose(x[:, c], (1.0 - dl1_embed.IMAGENET_MEAN[c]) / dl1_embed.IMAGENET_STD[c])
    rgb = np.zeros((1, 224, 224, 3), np.uint8)
    z = dl1_embed.to_model_input(rgb)
    assert z.shape == (1, 3, 224, 224)
    for c in range(3):
        assert np.allclose(z[:, c], -dl1_embed.IMAGENET_MEAN[c] / dl1_embed.IMAGENET_STD[c])


def test_t14_embed_is_cls_plus_patch_mean():
    """T14: the description is the CLS token, then the mean of the patch tokens."""
    class Stub:
        def forward_features(self, x):
            b = x.shape[0]
            return {"x_norm_clstoken": torch.arange(b * 384, dtype=torch.float32).reshape(b, 384),
                    "x_norm_patchtokens": torch.ones(b, 256, 384)}

    out = dl1_embed.embed(Stub(), torch.zeros(3, 3, 224, 224))
    assert out.shape == (3, 768)
    clstoken = torch.arange(3 * 384, dtype=torch.float32).reshape(3, 384)
    assert torch.allclose(out[:, :384], clstoken)
    assert torch.allclose(out[:, 384:], torch.ones(3, 384))


def test_t15_crop_to_input_chain():
    """T15: a single outline crop becomes one (1, 3, 224, 224) input whose channels hold the
    same picture. (They differ only by the per-channel ImageNet normalisation of T13/§5, so
    "identical" is checked after undoing it.)"""
    outline = np.zeros((224, 224), np.uint8)
    outline[50:150, 60:160] = 255
    x = dl1_embed.to_model_input(outline[None])
    assert x.shape == (1, 3, 224, 224)
    for c in range(3):
        unnormalised = x[0, c] * dl1_embed.IMAGENET_STD[c] + dl1_embed.IMAGENET_MEAN[c]
        assert np.allclose(unnormalised, outline / 255.0, atol=1e-6)
    assert np.allclose(x[0, 0, 50:150, 60:160], (1.0 - dl1_embed.IMAGENET_MEAN[0]) / dl1_embed.IMAGENET_STD[0])
