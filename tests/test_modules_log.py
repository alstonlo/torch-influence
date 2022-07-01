import functools
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from scipy import stats
from sklearn import linear_model
from torch import nn
from torch.utils import data

from torch_influence import BaseObjective, CGInfluenceModule, LiSSAInfluenceModule


# pylint: disable=too-few-public-methods

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@functools.lru_cache()
def load_binary_mnist():
    data_dir = str(pathlib.Path(__file__).parents[1] / "data")
    digits = [1, 3]

    bin_mnist = []
    for train in (True, False):
        mnist = torchvision.datasets.MNIST(data_dir, train=train, download=True, transform=T.ToTensor())
        X, Y = mnist.data, mnist.targets
        idxs = torch.logical_or(Y == digits[0], Y == digits[1])

        Y = Y[idxs].unsqueeze(1)
        Y = torch.where(Y == digits[1], 1.0, 0.0)

        X = X[idxs].flatten(start_dim=-2) / 255.0
        X = torch.concat([X, torch.ones_like(Y)], dim=-1)

        bin_mnist.append((X, Y))
    return bin_mnist


class LogRegObjective(BaseObjective):

    def __init__(self, l2):
        self.l2 = l2

    def train_outputs(self, model, batch):
        return torch.sigmoid(model(batch[0]))

    def train_loss_on_outputs(self, outputs, batch):
        return F.binary_cross_entropy(outputs, batch[1])

    def train_regularization(self, params):
        return 0.5 * self.l2 * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        return F.binary_cross_entropy_with_logits(model(batch[0]), batch[1])


class LogRegProblem:

    def __init__(self, module_cls, module_kwargs, batch_size=100, dtype=torch.float32, l2=2e-2):

        if module_cls is LiSSAInfluenceModule:
            lissa_defaults = {"repeat": 1, "depth": 5000, "scale": 10}
            lissa_defaults.update(module_kwargs)
            module_kwargs = dict(lissa_defaults)

        self.module_cls = module_cls
        self.damp = module_kwargs["damp"]
        self.dtype = dtype
        self.l2 = l2

        bin_mnist = tuple((X.to(dtype), Y.to(dtype)) for (X, Y) in load_binary_mnist())
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = bin_mnist

        self.n = self.X_train.shape[0]
        self.d = self.X_train.shape[1]

        train_set = data.TensorDataset(self.X_train[:, :-1], self.Y_train)
        test_set = data.TensorDataset(self.X_test[:, :-1], self.Y_test)

        self.model = self.fitted_model()

        self.module = module_cls(
            model=self.model,
            objective=LogRegObjective(l2),
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

        C = 1 / (n * self.l2 * 0.5)
        sk_model = linear_model.LogisticRegression(C=C, solver="lbfgs", tol=1e-10, max_iter=1000, fit_intercept=False)
        sk_model.fit(X, Y.squeeze(-1))
        params = torch.tensor(sk_model.coef_, dtype=self.dtype)

        pt_model = nn.Linear(self.d - 1, 1, bias=True)
        pt_model.weight = torch.nn.Parameter(params[:, :-1])
        pt_model.bias = torch.nn.Parameter(params[:, -1])
        pt_model = pt_model.to(device=DEVICE, dtype=self.dtype)
        return pt_model


@pytest.mark.parametrize(
    "idxs,train",
    [
        ([0], True),
        ([0], False),
        (list(range(101)), True),
        (list(range(101)), False)
    ],
)
def test_loss_grad_at_batch(idxs, train):
    problem = LogRegProblem(CGInfluenceModule, {"damp": 0.0})
    module = problem.module

    X = problem.X_train if train else problem.X_test
    Y = problem.Y_train if train else problem.Y_test
    X, Y = X[idxs].to(DEVICE), Y[idxs].to(DEVICE)

    w = torch.concat([problem.model.weight.T, problem.model.bias.unsqueeze(0)], dim=0).detach()

    analytic_grad = (X.T @ (torch.sigmoid(X @ w) - Y) / len(idxs)) + ((problem.l2 if train else 0.0) * w)
    analytic_grad = analytic_grad.squeeze(1)

    grad_fn = module.train_loss_grad if train else module.test_loss_grad
    module_grad = grad_fn(idxs)

    assert torch.allclose(analytic_grad, module_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "problem,test_idxs",
    [
        (LogRegProblem(CGInfluenceModule, {"damp": 0.0, "atol": 1e-8, "maxiter": 300}), [5, 100, 12]),
        (LogRegProblem(LiSSAInfluenceModule, {"damp": 0.0}), [5, 100, 12]),
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

    top_points = torch.argsort(if_scores).tolist()
    top_points = top_points[:20] + top_points[-20:]
    if_scores = if_scores[top_points]

    loo_scores = []
    for idx in top_points:
        loo_model = problem.fitted_model(remove_idx=idx)
        loo_params = nn.utils.parameters_to_vector(loo_model.parameters())
        score = module.objective.test_loss(loo_model, loo_params, test_batch) - base_loss
        loo_scores.append(score)
    loo_scores = torch.tensor(loo_scores)

    assert stats.pearsonr(loo_scores, if_scores)[0] > 0.9
    assert stats.spearmanr(loo_scores, if_scores)[0] > 0.9
