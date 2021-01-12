import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_linear_data(N=500, input_size=4, input_sd=1, output_size=1, output_sd=1):
    intercept = torch.randn(1, 1)
    w = torch.randn(input_size, output_size)
    X = torch.randn(N, input_size) * input_sd
    y_mean = intercept + torch.matmul(X, w)
    y = y_mean + torch.randn(N, 1) * output_sd

    return X, y, w, intercept, y_mean


def gd(model, X, y, epochs):
    """Train neural network using Adam. """
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    criterion = nn.MSELoss()

    for i in range(epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5000 == 0:
            print(f"Done: {i + 1} / {epochs}")

    return model, losses


def generate_samples(dict_of_samplers, dict_of_algo, X, y, samples, seed=1234):
    """
    Generates samples from a number of samplers

    Parameters:
        dict_of_samplers - dictionary of RegressionSampler classes

        dict_of_algo - dictionary indicating which algorithm to use (rw or lg), keys must match dict_of_samplers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for name, sampler in dict_of_samplers.items():
        if dict_of_algo[name] == "rw":
            for i in range(samples):
                sampler.rw_mcmc(X, y)
                if (i + 1) % 1000 == 0:
                    print(f"RW sampling for {name}, done: {i + 1} / {samples}")

        elif dict_of_algo[name] == "lg":
            for i in range(samples):
                sampler.lg_mcmc(X, y)
                if (i + 1) % 1000 == 0:
                    print(f"LG sampling for {name}, done: {i + 1} / {samples}")

        else:
            for i in range(samples):
                sampler.mcmc(X, y)
                if (i + 1) % 1000 == 0:
                    print(f"MCMC sampling for {name}, done: {i + 1} / {samples}")


def display_results(dict_of_samplers, burn_in, X, y, base_model):
    """
    Plots sample results from a number of samplers. This includes final acceptance rate/ loss, rolling average
    acceptance rate, tausq trace and distribution, parameter (weights and biases) trace and distribution and
    prediction (train data) trace and distribution

    Parameters:
        dict_of_samplers - dictionary of RegressionSampler classes that already have sample generated (ie. using the
        generate_samples function)

    """
    # print losses, final acceptance rates
    for name, sampler in dict_of_samplers.items():
        final_acceptance_rate = sum([w[-1] for w in sampler.container.data]) / len(sampler.container.data)
        print(f"Final loss, RMSE, {name}: {sampler.container.data[-1][3] ** 0.5}")
        print(f"Final acceptance rate, {name}: {final_acceptance_rate}")

    # plot loss per sample
    plt.figure()
    plt.title("Loss per sample, RMSE")
    for name, sampler in dict_of_samplers.items():
        plt.plot(np.sqrt([w[3] for w in sampler.container.data][burn_in:]), label=name)
    plt.legend()
    plt.show()

    # plot rolling average acceptance rate
    plt.figure()
    plt.title("Rolling average acceptance rate")
    for name, sampler in dict_of_samplers.items():
        rolling_avg = pd.Series([w[-1] for w in sampler.container.data]).rolling(500).mean().dropna()
        plt.plot(rolling_avg, label=name)
    plt.legend()
    plt.show()

    # tausq trace plot and distribution
    plt.figure()
    plt.title("Trace plot: tausq")
    for name, sampler in dict_of_samplers.items():
        plt.plot([w[1] for w in sampler.container.data], label=name)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Distribution: tausq")
    for name, sampler in dict_of_samplers.items():
        plt.hist([w[1] for w in sampler.container.data][burn_in:],
                 label=name, density=True, alpha=0.5, bins=20)
    plt.legend()
    plt.show()

    # parameter trace plot and distribution
    for i in range(3):
        param_len = len(list(base_model.parameters()))
        idx1 = np.random.choice(range(param_len))

        layer_shape = list(base_model.parameters())[idx1].shape
        idx2 = np.random.choice(range(layer_shape[0]))

        if len(layer_shape) > 1:
            idx3 = np.random.choice(range(layer_shape[1]))

            plt.figure()
            plt.title(f"Trace plot: coefficient {idx1}, {idx2}, {idx3}")
            for name, sampler in dict_of_samplers.items():
                plt.plot([w[0][idx1][idx2, idx3].item() for w in sampler.container.data], label=name)
            plt.legend()
            plt.show()

            plt.figure()
            plt.title(f"Distribution: coefficient {idx1}, {idx2}, {idx3}")
            for name, sampler in dict_of_samplers.items():
                plt.hist([w[0][idx1][idx2, idx3].item() for w in sampler.container.data][burn_in:],
                         label=name, density=True, alpha=0.5, bins=20)
            plt.legend()
            plt.show()

        else:
            plt.figure()
            plt.title(f"Trace plot: coefficient {idx1}, {idx2}")
            for name, sampler in dict_of_samplers.items():
                plt.plot([w[0][idx1][idx2].item() for w in sampler.container.data], label=name)
            plt.legend()
            plt.show()

            plt.figure()
            plt.title(f"Distribution: coefficient {idx1}, {idx2}")
            for name, sampler in dict_of_samplers.items():
                plt.hist([w[0][idx1][idx2].item() for w in sampler.container.data][burn_in:],
                         label=name, density=True, alpha=0.5, bins=20)
            plt.legend()
            plt.show()

    # prediction trace plot and distribution
    for i in range(5):
        idx = np.random.choice(len(y))

        plt.figure()
        plt.title(f"Trace plot: predction {idx}")
        for name, sampler in dict_of_samplers.items():
            plt.plot([w[2][idx].item() for w in sampler.container.data], label=name)
        plt.legend()
        plt.show()

        plt.figure()
        plt.title(f"Trace plot: predction {idx}")
        for name, sampler in dict_of_samplers.items():
            plt.hist([w[2][idx].item() for w in sampler.container.data][burn_in:],
                     label=name, density=True, alpha=0.5, bins=20)
        plt.legend()
        plt.show()