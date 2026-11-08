import torch

from eikonal_pinn.model import EikonalPINN, MLP


def test_forward_and_residual_shapes():
    torch.manual_seed(0)
    m = EikonalPINN()
    for n in (1, 7, 100):
        xy = torch.rand(n, 2)
        T, V = m(xy)
        assert T.shape == (n, 1) and V.shape == (n, 1)
        assert torch.all(V > 0) and torch.all(V < 1.5)
        f_T, f_V, T2, V2 = m.residual(xy)
        assert f_T.shape == (n, 1) and f_V.shape == (n, 1)
        assert T2.shape == (n, 1) and V2.shape == (n, 1)


def test_architecture_and_init():
    m = EikonalPINN()
    assert [l.in_features for l in m.net_T.linears] == [2, 20, 20, 20, 20, 20]
    assert [l.out_features for l in m.net_V.linears] == [5, 5, 5, 5, 1]
    assert all(torch.all(l.bias == 0) for l in m.net_T.linears)
    assert len(m.net_T.weight_matrices()) == 6
    w = MLP([2, 20, 20, 1], "relu")
    assert w.act_name == "relu"
