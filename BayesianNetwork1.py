import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class BayesianRegressionWrapper(nn.Module):
    """Wrapper to extend functionality of a Pytorch regression network into a Bayesian network. """

    def __init__(self, base_model, improper=True, sigmasq=25.0, nu1=0.0, nu2=0.0):
        super().__init__()
        self.base_model = base_model
        self.improper = improper
        self.sigmasq = sigmasq
        self.nu1 = nu1
        self.nu2 = nu2
        self.tausq = np.exp(np.random.normal())

    def log_llk(self, X, y):
        y_num = y.numel()
        y_pred = self(X)
        part1 = -0.5 * y_num * np.log(2 * np.pi * self.tausq)
        part2 = -0.5 * y_num * nn.MSELoss()(y_pred, y) / self.tausq
        return part1 + part2

    def log_prior(self):
        n_params = sum([w_.numel() for w_ in self.parameters()])
        part0 = -0.5 * n_params * np.log(self.sigmasq)
        part1 = -0.5 * sum([torch.sum(w_ ** 2) for w_ in list(self.parameters())]) / self.sigmasq
        part2 = -(self.nu1 + 1) * np.log(self.tausq) - self.nu2 / self.tausq
        return part0 + part1 + part2

    def set_params(self, w):
        self.base_model.load_state_dict(w)

    def set_params_from_list(self, new_w):
        for w_, new_w_ in zip(self.parameters(), new_w):
            w_.data = new_w_.data

    def forward(self, i):
        return self.base_model(i)

    def parameters(self):
        return self.base_model.parameters()


class RegressionSampler:
    """
    Class to generate samples from a Bayesian neural network. Implements RW-MCMC and the LG-MCMC detailed in
    Chandra, Azizi & Cripps, 2017.
    """
    def __init__(self, bayesian_model, X_train, y_train, lr=0.01, prop_sd=0.05, tausq_prop_sd=0.20):
        self.model = bayesian_model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.prop_sd = prop_sd
        self.tausq_prop_sd = tausq_prop_sd

        self.container = SampleContainerLite()

        with torch.no_grad():
            y_pred = self.model(X_train)
            loss = nn.MSELoss()(y_pred, y_train)
            accept = 0

        self.container.append(
            deepcopy(self.model.base_model.state_dict()), deepcopy(self.model.tausq),
            deepcopy(y_pred), deepcopy(loss.item()), accept)

    def rw_mcmc(self, X_train, y_train): # verified
        w_k = deepcopy(self.model.base_model.state_dict())
        tausq_k = self.model.tausq

        tausq_p = self.propose_tausq(tausq_k)
        w_p = self.propose_w(w_k)

        with torch.no_grad():
            log_post_p = self.log_posterior(X_train, y_train, w_p, tausq_p)
            log_post_k = self.log_posterior(X_train, y_train, w_k, tausq_k)
            log_alpha = log_post_p - log_post_k

        if log_alpha > np.log(np.random.rand()):
            self.model.set_params(w_p)
            self.model.tausq = tausq_p
            accept = 1
        else:
            self.model.set_params(w_k)
            self.model.tausq = tausq_k
            accept = 0

        with torch.no_grad():
            y_pred = self.model(X_train)
            loss = nn.MSELoss()(y_pred, y_train)

        self.container.append(
            deepcopy(self.model.base_model.state_dict()), deepcopy(self.model.tausq),
            deepcopy(y_pred), deepcopy(loss.item()), accept)

    def lg_mcmc(self, X_train, y_train): # verfied
        w_k = deepcopy(self.model.base_model.state_dict())
        tausq_k = self.model.tausq

        tausq_p = self.propose_tausq(tausq_k)
        w_k_bar = self.compute_gradient_update(X_train, y_train, deepcopy(w_k), self.model.tausq)
        w_p = self.propose_w(w_k_bar)
        w_p_bar = self.compute_gradient_update(X_train, y_train, deepcopy(w_p), self.model.tausq)

        with torch.no_grad():
            log_post_p = self.log_posterior(X_train, y_train, w_p, tausq_p)
            log_post_k = self.log_posterior(X_train, y_train, w_k, tausq_k)
            log_prop_diff = self.log_prop_diff(w_p, w_p_bar, w_k, w_k_bar)
            log_alpha = log_post_p - log_post_k + log_prop_diff

        if log_alpha > np.log(np.random.rand()):
            self.model.set_params(w_p)
            self.model.tausq = tausq_p
            accept = 1
        else:
            self.model.set_params(w_k)
            self.model.tausq = tausq_k
            accept = 0

        with torch.no_grad():
            y_pred = self.model(X_train)
            loss = nn.MSELoss()(y_pred, y_train)

        self.container.append(
            deepcopy(self.model.base_model.state_dict()), deepcopy(self.model.tausq),
            deepcopy(y_pred), deepcopy(loss.item()), accept)

        return loss.item()

    def compute_gradient_update(self, X, y, w, tausq):
        self.model.set_params(deepcopy(w))
        self.model.tausq = tausq

        y_pred = self.model(X)
        mse = nn.MSELoss()(y_pred, y)
        self.optimizer.zero_grad()
        mse.backward()
        self.optimizer.step()

        w_bar = deepcopy(self.model.base_model.state_dict())
        return w_bar

    @torch.no_grad()
    def log_prop_diff(self, w_p, w_p_bar, w_k, w_k_bar):
        """Computes difference in log proposal probabilities. """
        log_num = sum([torch.sum((x - y) ** 2) for x, y in zip(w_p_bar.values(), w_k.values())])
        log_den = sum([torch.sum((x - y) ** 2) for x, y in zip(w_p.values(), w_k_bar.values())])
        return -0.5 * (log_num - log_den) / self.prop_sd ** 2

    @torch.no_grad()
    def propose_w(self, w):
        w_p = {}
        for key in w.keys():
            w_p[key] = deepcopy(w[key]) + torch.zeros(w[key].size()).normal_(mean=0, std=self.prop_sd)
        return w_p

    def propose_tausq(self, tausq):
        tausq_p = np.exp(np.log(tausq) + np.random.normal(scale=self.tausq_prop_sd))
        return tausq_p

    @torch.no_grad()
    def log_posterior(self, X, y, w, tausq):
        self.model.set_params(deepcopy(w))
        self.model.tausq = tausq
        return self.model.log_llk(X, y) + self.model.log_prior()


class SampleContainerLite:
    """Container to store samples (ie. weights and biases, tausq, train predictions, train loss and accept indicator"""
    def __init__(self):
        self.data = []

    def append(self, w, tausq, y_pred_train, loss, accept):
        w = [w_.detach() for w_ in w.values()]
        y_pred_train = y_pred_train.detach()
        self.data.append([w, tausq, y_pred_train, loss, accept])


