import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn import datasets
from torch import nn
from torch.utils import data

from torch_influence import AutogradInfluenceModule, BaseObjective, CGInfluenceModule, LiSSAInfluenceModule


# pylint: disable=too-few-public-methods

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinRegObjective(BaseObjective):

    def __init__(self, l2):
        self.l2 = l2

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.mse_loss(outputs, batch[1])

    def train_regularization(self, params):
        return 0.5 * self.l2 * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        return F.mse_loss(model(batch[0]), batch[1])


class LinRegProblem:

    def __init__(self, module_cls, module_kwargs, batch_size=7, dtype=torch.float32, l2=1e-2):

        if module_cls is LiSSAInfluenceModule:
            lissa_defaults = {"repeat": 20, "depth": 200, "scale": 50}
            lissa_defaults.update(module_kwargs)
            module_kwargs = dict(lissa_defaults)

        self.module_cls = module_cls
        self.damp = module_kwargs["damp"]
        self.dtype = dtype
        self.l2 = l2

        self.n = 100
        self.d = 11

        X, Y = datasets.make_regression(
            n_samples=(self.n + 10),
            n_features=(self.d - 1),
            random_state=12345,
            n_informative=(self.d - 1),
            noise=1
        )

        Y = torch.tensor(Y, dtype=dtype).unsqueeze(1)
        X = torch.tensor(X, dtype=dtype)
        X = torch.cat([X, torch.ones_like(Y)], dim=-1)

        X_train, X_test = X[:self.n], X[self.n:]
        Y_train, Y_test = Y[:self.n], Y[self.n:]

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        train_set = data.TensorDataset(X_train[:, :-1], Y_train)
        test_set = data.TensorDataset(X_test[:, :-1], Y_test)

        self.model = self.fitted_model()

        self.module = module_cls(
            model=self.model,
            objective=LinRegObjective(l2),
            train_loader=data.DataLoader(train_set, batch_size=batch_size),
            test_loader=data.DataLoader(test_set, batch_size=batch_size),
            device=DEVICE,
            **module_kwargs
        )

    def fitted_model(self, remove_idx=None):
        X, Y = self.X_train, self.Y_train
        if remove_idx is not None:
            X = np.delete(X, remove_idx, axis=0)
            Y = np.delete(Y, remove_idx, axis=0)
        n = X.shape[0]

        params = torch.inverse(X.T @ X + (n * self.l2 * 0.5) * torch.eye(self.d)) @ X.T @ Y

        pt_model = nn.Linear(self.d - 1, 1, bias=True)
        pt_model.weight = torch.nn.Parameter(params[:-1].T)
        pt_model.bias = torch.nn.Parameter(params[-1])
        pt_model = pt_model.to(device=DEVICE, dtype=self.dtype)
        return pt_model


@pytest.mark.parametrize(
    "idxs,train",
    [
        ([0], True),
        ([0], False),
        (list(range(8)), True),
        (list(range(8)), False)
    ],
)
def test_loss_grad_at_batch(idxs, train):
    problem = LinRegProblem(CGInfluenceModule, {"damp": 0.0})
    module = problem.module

    X = problem.X_train if train else problem.X_test
    Y = problem.Y_train if train else problem.Y_test
    X, Y = X[idxs].to(DEVICE), Y[idxs].to(DEVICE)
    w = torch.concat([problem.model.weight.T, problem.model.bias.unsqueeze(0)], dim=0).detach()

    analytic_grad = (2 * X.T @ (X @ w - Y) / len(idxs)) + ((problem.l2 if train else 0.0) * w)
    analytic_grad = analytic_grad.squeeze(1)

    grad_fn = module.train_loss_grad if train else module.test_loss_grad
    module_grad = grad_fn(idxs)

    assert torch.allclose(analytic_grad, module_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "problem",
    [
        LinRegProblem(AutogradInfluenceModule, {"damp": 0.02, "check_eigvals": True}),
        LinRegProblem(CGInfluenceModule, {"damp": 0.02, "atol": 0.0}),
        LinRegProblem(CGInfluenceModule, {"damp": 0.02, "gnh": True, "atol": 0.0}),
        LinRegProblem(LiSSAInfluenceModule, {"damp": 0.02}, batch_size=64),
        LinRegProblem(LiSSAInfluenceModule, {"damp": 0.02, "gnh": True}, batch_size=64),
    ],
)
def test_inverse_hvp(problem):
    X = problem.X_train.to(DEVICE)
    hess = ((2 / problem.n) * X.T @ X) + (problem.damp + problem.l2) * torch.eye(problem.d, device=DEVICE)

    torch.manual_seed(12345)
    for _ in range(3):
        vec = torch.randn(problem.d, device=DEVICE, dtype=problem.dtype)
        analytic_ihvp = torch.inverse(hess) @ vec
        module_ihvp = problem.module.inverse_hvp(vec)

        tol = 0.015 if (problem.module_cls is LiSSAInfluenceModule) else 1e-3
        assert torch.allclose(analytic_ihvp, module_ihvp, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "problem,test_idxs",
    [
        (LinRegProblem(AutogradInfluenceModule, {"damp": 0.0}), list(range(3, 9))),
        (LinRegProblem(CGInfluenceModule, {"damp": 0.0, "atol": 0.0}), list(range(3, 9))),
        (LinRegProblem(LiSSAInfluenceModule, {"damp": 0.0}, batch_size=64), list(range(3, 9))),
    ],
)
def test_influence_scores(problem, test_idxs):
    module = problem.module

    test_batch = module.test_loader.dataset[test_idxs]
    test_batch = test_batch[0].to(DEVICE), test_batch[1].to(DEVICE)

    base_params = nn.utils.parameters_to_vector(module.model.parameters())
    base_loss = module.objective.test_loss(module.model, base_params, test_batch)

    train_idxs = list(range(problem.n))
    if_scores = module.influences(train_idxs=train_idxs, test_idxs=test_idxs).cpu()

    loo_scores = []
    for idx in train_idxs:
        loo_model = problem.fitted_model(remove_idx=idx)
        loo_params = nn.utils.parameters_to_vector(loo_model.parameters())
        score = module.objective.test_loss(loo_model, loo_params, test_batch) - base_loss
        loo_scores.append(score)
    loo_scores = torch.tensor(loo_scores)

    assert stats.pearsonr(loo_scores, if_scores)[0] > 0.9
    assert stats.spearmanr(loo_scores, if_scores)[0] > 0.9
