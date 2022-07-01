import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data

from torch_influence import AutogradInfluenceModule, BaseObjective, CGInfluenceModule


# pylint: disable=redefined-outer-name,protected-access

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_mock_module(module_cls, batch_size, dtype):
    torch.manual_seed(100)

    mlp = nn.Sequential(
        nn.Linear(10, 32),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.Dropout(p=0.5),
        nn.Linear(32, 32),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.Dropout(p=0.5),
        nn.Linear(32, 1),
    ).to(dtype)

    class MockObjective(BaseObjective):

        def train_outputs(self, model, batch):
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            return F.mse_loss(outputs, batch[1])

        def train_regularization(self, params):
            return params.norm() ** 2.0

        def test_loss(self, model, params, batch):
            return F.mse_loss(model(batch[0]), batch[1])

    dataset = data.TensorDataset(torch.randn((50, 10), dtype=dtype), torch.randn((50, 1), dtype=dtype))
    train_loader = data.DataLoader(dataset, batch_size=batch_size)
    test_loader = data.DataLoader(dataset, batch_size=batch_size)

    return module_cls(
        model=mlp,
        objective=MockObjective(),
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        damp=0.01,
    )


@pytest.fixture()
def module_f32_b7():
    return make_mock_module(CGInfluenceModule, batch_size=7, dtype=torch.float32)


@pytest.fixture()
def module_f32_bf():
    return make_mock_module(CGInfluenceModule, batch_size=50, dtype=torch.float32)


@pytest.fixture()
def module_f64():
    return make_mock_module(AutogradInfluenceModule, batch_size=7, dtype=torch.float64)


def test_param_reshaping(module_f32_b7):
    params = module_f32_b7._model_params(with_names=False)

    flat_params = module_f32_b7._flatten_params_like(params)
    assert flat_params.shape == (1569,)

    zeros = torch.zeros_like(flat_params)
    zeros = module_f32_b7._reshape_like_params(zeros)
    assert isinstance(zeros, tuple) and len(zeros) == 10


def test_make_functional(module_f32_b7):
    parmas1 = module_f32_b7._model_params(with_names=False)
    params2 = module_f32_b7._model_make_functional()

    assert len(parmas1) == len(params2)
    assert all(torch.equal(*ps) for ps in zip(parmas1, params2))

    with torch.no_grad():
        module_f32_b7._model_reinsert_params(params2, register=True)
    params3 = module_f32_b7._model_params(with_names=False)

    assert len(parmas1) == len(params3)
    assert all(torch.equal(*ps) for ps in zip(parmas1, params3))


def test_precision(module_f64):
    module = module_f64
    assert module.train_loss_grad([1, 2]).dtype == torch.float64
    assert module.test_loss_grad([1, 20]).dtype == torch.float64

    vec = torch.randn(1569, device=DEVICE, dtype=torch.float64)
    assert module.inverse_hvp(vec).dtype == torch.float64


def test_index_error(module_f32_b7):
    with pytest.raises(IndexError):
        module_f32_b7.train_loss_grad([-1])
    with pytest.raises(IndexError):
        module_f32_b7.train_loss_grad([50])

    with pytest.raises(ValueError):
        module_f32_b7.stest([2, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "loader_kwargs,batch_sizes",
    [
        ({}, [7] * 7 + [1]),
        ({"batch_size": 10}, [10] * 5),
        ({"sample_n_batches": 5}, [7] * 5),
        ({"batch_size": 1, "subset": [1, 2, 3]}, [1] * 3),
        ({"subset": list(range(12))}, [7, 5]),
    ],
)
def test_loader_wrapper(module_f32_b7, loader_kwargs, batch_sizes):
    for i, (batch, batch_size) in enumerate(module_f32_b7._loader_wrapper(train=True, **loader_kwargs)):
        assert batch_size == batch_sizes[i]
        assert batch_size == batch[0].shape[0]


def test_loss_grad_at_batch(module_f32_b7, module_f32_bf):
    g1 = module_f32_b7.train_loss_grad(list(range(20)))
    g2 = module_f32_bf.train_loss_grad(list(range(20)))
    assert torch.allclose(g1, g2)

    g31 = module_f32_b7.train_loss_grad(list(range(11)))
    g32 = module_f32_b7.train_loss_grad(list(range(11, 20)))
    g3 = (11 / 20) * g31 + (9 / 20) * g32
    assert torch.allclose(g2, g3)
