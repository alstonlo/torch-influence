.. torch-influence documentation master file, created by
   sphinx-quickstart on Wed Jun 29 20:51:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to torch-influence's API Reference!
===========================================

.. contents:: Table of Contents
   :depth: 2
   :local:

.. currentmodule:: torch_influence

Base Modules
------------------------

.. autoclass:: BaseInfluenceModule
   :members:

.. autoclass:: BaseObjective
   :members:


Influence Modules
------------------------

torch-influence provides three subclasses of :class:`BaseInfluenceModule` out-of-the-box.
Each subclass differs only in how the abstract function :meth:`BaseInfluenceModule.inverse_hvp()`
is implemented. We refer readers to the original influence function
paper_ (Koh & Liang, 2017) for further details.

.. _paper: https://arxiv.org/abs/1703.04730

.. autoclass:: AutogradInfluenceModule
   :show-inheritance:

.. autoclass:: CGInfluenceModule
   :show-inheritance:

.. autoclass:: LiSSAInfluenceModule
   :show-inheritance:
