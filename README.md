<div align="center">    

# torch-influence

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/alstonlo/torch-influence)
[![Read the Docs](https://img.shields.io/readthedocs/torch-influence)](https://torch-influence.readthedocs.io/en/latest/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)

</div>

![](examples/dogfish_influences.png)

torch-influence is a PyTorch implementation of influence functions, a classical
technique from robust statistics that estimates the effect of removing a single training data point on a model’s
learned parameters. In their seminal paper _Understanding Black-box Predictions via Influence Functions_
([paper](https://arxiv.org/abs/1703.04730)),
Koh & Liang (2017) first co-opted influence functions to the domain of machine learning. Since then,
influence functions have been applied on a variety of machine learning tasks,
including explaining model predictions, dataset relabelling and reweighing,
data poisoning, increasing model fairness, and data augmentation.

This library aims to be simple and minimal. In addition, it fixes a few errors found in some of the existing
implementations of influence functions.

The code is supplement to the paper [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/abs/2209.05364). Furthermore, the Jax implementation can be found at [here](https://github.com/pomonam/jax-influence).

______________________________________________________________________

## Installation

Pip from source:

```bash
git clone https://github.com/alstonlo/torch-influence
 
cd torch_influence
pip install -e .   
 ```

______________________________________________________________________

## Quickstart

### Overview

In order to use torch-influence, the first step is to subclass its `BaseInfluenceModule` class and implement its
single abstract method `BaseInfluenceModule.inverse_hvp()`. This method computes inverse Hessian-vector products (iHVPs), 
which is an important but costly step in influence function computation. Conveniently, torch-influence provides three 
subclasses out-of-the-box:


<div align="center">
 
| Subclass  | Method of iHVP computation |
| ------------- | ------------- |
| `AutogradInfluenceModule`  | Direct computation and inversion of the Hessian with `torch.autograd`  |
| `CGInfluenceModule`  | Truncated Conjugate Gradients (Martens et al., 2010) ([paper](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)) |
| `LiSSAInfluenceModule`  | Linear time Stochastic Second-Order Algorithm (Agarwal et al., 2016) ([paper](https://arxiv.org/abs/1602.03943)) |

</div>

The next step is to subclass `BaseObjective` and implement its four abstract methods. 
The `BaseObjective` class serves as an adapter that holds project-specific information about how 
training and test losses are computed. 
All of `BaseInfluenceModule` and its three subclasses require an implementation of `BaseObjective` to be passed through its constructor.
The following is a sample subclass for an $L_2$-regularized classification model:


```python
import torch
import torch.nn.functional as F
from torch_influence import BaseObjective


class MyObjective(BaseObjective):

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.cross_entropy(outputs, batch[1])  # mean reduction required

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    # training loss by default taken to be 
    # train_loss_on_outputs + train_regularization

    def test_loss(self, model, params, batch):
        return F.cross_entropy(model(batch[0]), batch[1])  # no regularization in test loss
```

Finally, all that is left is to piece everything together. 
After instantiating a subclass of `BaseInfluenceModule`, 
influence scores can then be computed through the `BaseInfluenceModule.influences()` method.
For example:

```python 
from torch_influence import AutogradInfluenceModule
   

module = AutogradInfluenceModule(
    model=model,
    objective=MyObjective(),  
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    damp=0.001
)

# influence scores of training points 1, 2, and 3 on test point 0
scores = module.influences([1, 2, 3], [0])
```


For more details, we refer users to the [API Reference](https://torch-influence.readthedocs.io/en/latest/).



### Dogfish 

The `examples/` directory contains a more complete example, which finetunes the topmost
layer of a pretrained Inceptionv3 network on the Dogfish dataset (Koh & Liang, 2017). Then, it 
uses influence functions to find the most helpful and harmful training images,
with respect to a couple of test images. To run the example, please download and extract
the Dogfish dataset ([CodaLab](https://worksheets.codalab.org/bundles/0x550cd344825049bdbb865b887381823c))
into the `examples/` folder and execute the following:


```bash
# install dependencies
pip install -e .[dev]  

cd examples/

# train model and analyze influence scores
python analyze_dogfish.py  
```  

______________________________________________________________________

## Contributors

- [Alston Lo](https://github.com/alstonlo)
- [Juhan Bae](https://www.juhanbae.com/)

