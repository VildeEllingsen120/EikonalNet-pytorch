import torch

from eikonal_pinn.model import EikonalPINN


def _patched(residual_form="tf"):
    torch.manual_seed(0)
    m = EikonalPINN(residual_form=residual_form).double()
    # monkeypatch forward: T = x^2 + y, V = 2 + x
    def fwd(xy):
        x, y = xy[:, :1], xy[:, 1:]
        return x**2 + y, 2.0 + x
    m.forward = fwd
    return m


def test_gradients_and_create_graph():
    m = _patched()
    xy = torch.rand(64, 2, dtype=torch.float64).requires_grad_(True)
    f_T, f_V, T, V = m.residual(xy)
    x = xy[:, :1]
    grad_T = m._grad(T, xy)
    grad_V = m._grad(V, xy)
    assert torch.allclose(grad_T[:, :1], 2 * x, atol=1e-6)
    assert torch.allclose(grad_T[:, 1:], torch.ones_like(x), atol=1e-6)
    assert torch.allclose(grad_V[:, :1], torch.ones_like(x), atol=1e-6)
    assert torch.allclose(grad_V[:, 1:], torch.zeros_like(x), atol=1e-6)
    # residual values
    nT = torch.sqrt(4 * x**2 + 1)
    assert torch.allclose(f_T, nT - 1 / (2 + x), atol=1e-6)
    assert torch.allclose(f_V, torch.ones_like(x), atol=1e-6)
    # f_V is a constant here (grad V = (1, 0) has no graph), so check it with the real net below
    assert f_T.requires_grad


def test_paper_residual_form():
    m = _patched("paper")
    xy = torch.rand(16, 2, dtype=torch.float64).requires_grad_(True)
    f_T, _, _, _ = m.residual(xy)
    x = xy[:, :1]
    assert torch.allclose(f_T, (2 + x) * torch.sqrt(4 * x**2 + 1) - 1, atol=1e-6)


def test_residual_depends_on_parameters():
    """Second-order path: d(residual loss)/d(theta) must be nonzero (create_graph on)."""
    torch.manual_seed(0)
    m = EikonalPINN()
    xy = torch.rand(32, 2)
    f_T, f_V, _, _ = m.residual(xy)
    assert f_T.requires_grad and f_V.requires_grad
    loss = (f_T**2).mean() + (f_V**2).mean()
    grads = torch.autograd.grad(loss, list(m.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
