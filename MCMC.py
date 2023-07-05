import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import random
import math
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, norm
import hamiltorch
import torch.distributions as dist
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
device = "cuda" if torch.cuda.is_available() else "cpu"

def likelihood(x, y, theta, std = 1):
  mean = theta * x
  f = (1 / (std * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-(y - mean)**2 / (2 * std**2))
  return torch.mean(torch.diag(f))

def samples_list(x, y, min, max, step, dist = "normal"):
  list_ = []
  init = random.uniform(min, max)
  mean = init
  mybar = st.progress(0, text = "Learning...")
  iterations = 200000
  for iteration in trange(iterations):
    init = random.normalvariate(mean, step)
    if dist == "normal":
      alpha = float(likelihood(x, y, init)/ likelihood(x, y, mean)) # Prior is assumed uniform
      if math.isnan(alpha):
        alpha = 1
    u = random.uniform(0,1)
    if (alpha > u):
      list_.append(init)
      mean = init
    if iteration < iterations * 0.8:
        # st.write(iteration, iterations)
        mybar.progress(float(iteration)/iterations, text = "Learning...")
    else:
        if iteration < iterations - 1:
            mybar.progress(float(iteration)/iterations, text = "Almost There...")
        else:
            mybar.progress(float(iteration)/iterations, text = "Done!")
  return list_

def rejection_sampling(range_x, method, histplot = False, show_original = False):
  x = torch.linspace(range_x[0], range_x[1], 1000)
  if method == "neg-exp":
    y = torch.exp(-x)
  elif method == "sin":
    y = 0.5 * torch.sin(x) + 0.5
  st.header("Original Distribution")
  plt.plot(x, y)
  plt.show()
  st.pyplot()
  
  mybar = st.progress(0, text = "Learning...")
  iterations = 200000
  list = []
  for iteration in range(iterations):
    a = random.uniform(range_x[0], range_x[1])
    b = random.uniform(float(torch.min(y)), float(torch.max(y)))
    if method == "neg-exp":
      if (np.exp(-a) > b):
        list.append(a)
    if method == "sin":
      if (0.5 * np.sin(a) + 0.5 > b):
        list.append(a)
    if iteration < iterations * 0.8:
        # st.write(iteration, iterations)
        mybar.progress(float(iteration)/iterations, text = "Learning...")
    else:
        if iteration < iterations - 1:
            mybar.progress(float(iteration)/iterations, text = "Almost There...")
        else:
            mybar.progress(float(iteration)/iterations, text = "Done!")

  st.header("Sample Distribution")
  if histplot:
    sns.histplot(list[10000: ], stat = "probability", bins = 10)
  sns.kdeplot(list[10000:])
  # plt.show()
  if show_original:
    plt.plot(x, y)
    plt.show()
  st.pyplot()

def bivariate_normal_pdf(x, y, mean, covariance):
    
    cov_det = covariance[0, 0] * covariance[1, 1] - covariance[0, 1] * covariance[1, 0]
    cov_inv = np.array([[covariance[1, 1], -covariance[0, 1]], [-covariance[1, 0], covariance[0, 0]]]) / cov_det

    X = np.array([x, y])
    mu = np.array(mean)

    exponent = -0.5 * np.dot(np.dot((X - mu).T, cov_inv), (X - mu))
    denominator = 2 * np.pi * np.sqrt(cov_det)

    pdf = (1 / denominator) * np.exp(exponent)

    return pdf

def log_prob(theta):
  for i in range(theta.squeeze().shape[0]):
    if i == 0:
      a = theta.squeeze()[i] * pow(x, theta.squeeze().shape[0] - 1 - i)
    else:
      a += theta.squeeze()[i] * pow(x, theta.squeeze().shape[0] - 1 - i)
  return dist.Normal(a, 1).log_prob(y).sum()

def log_prior(theta):
  return dist.Normal(0, 1).log_prob(theta).sum()

def log_posterior(theta):
  return log_prob(theta) + log_prior(theta)

def polynomial_samples_list(x, y, min, max, step):
  init = (torch.rand(degree + 1,) * (max - min) + min).to(device)
  list_ = init.unsqueeze(0)
  init0 = init
  mybar = st.progress(0, text = "")
  for i in trange(10000):
    init = dist.Normal(init0, 1).sample((1,))
    alpha = float((log_posterior(init) - log_posterior(init0)).exp())
    u = random.uniform(0,1)
    if (alpha > u):
      list_ = torch.cat((list_, init.squeeze().unsqueeze(0)), axis = 0)
      init0 = init
    mybar.progress(float(i / 10000))
  return list_

st.header("MCMC")

options = ["Linear Regression", "Uni-Variate Rejection Sampling", "Bi-Variate Rejection Sampling", "Polynomial Regression"]
task = st.multiselect("Select Task", options, max_selections = 1)
 
if "Linear Regression" in task:
    x, y = make_regression(n_samples = 100, n_features = 1)
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    st.header("Generated Line")
    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    st.pyplot()

    play = st.button("Play", type = "primary",use_container_width = True)
    if play:
        list_ = samples_list(x, y, -1000, 1000, 5)[10000:]

        st.header("KDE Plot of Samples")
        sns.kdeplot(list_)
        st.pyplot()

        list_ = torch.tensor(list_).reshape(-1,1)
        list_ = list_.to(device)

        X = torch.linspace(float(torch.min(x)), float(torch.max(x)), 1000).to(device)
        mean_pred = torch.mean(list_ @ X.reshape(-1,1).T.type(torch.float), axis = 0)
        std_pred = torch.std(list_ @ X.reshape(-1,1).T.type(torch.float), axis = 0)

        st.header("Predictions")
        plt.plot(X.cpu().detach(), mean_pred.cpu().detach(), label = "Predictions")
        plt.fill_between(X.ravel().cpu().detach(), (mean_pred - std_pred).ravel().cpu().detach(), (mean_pred + std_pred).ravel().cpu().detach(), alpha = 0.2, color = "gray")
        plt.plot(x.cpu().detach(),y.cpu().detach(), label = "True")
        plt.legend(loc = "best")
        st.pyplot()

if "Uni-Variate Rejection Sampling" in task:
    dist_opt = ["neg-exp", "sin"]
    dist = st.radio("Select distribution", dist_opt)
    histplot = st.radio("Show Histogram: ", [True, False])
    show_original = st.radio("Show Original Distribution: ", [True, False])
    play = st.button("Play", type = "primary",use_container_width = True)
    if play:
        if dist == "neg-exp":
            rejection_sampling((0,10), "neg-exp", histplot = histplot, show_original = show_original)
        if dist == "sin":
            rejection_sampling((0,10), "sin", histplot = histplot, show_original = show_original)

if "Bi-Variate Rejection Sampling" in task:
    mean1 = st.slider("Mean1", -10.00, 10.00, step = 0.01)
    mean2 = st.slider("Mean2", -10.00, 10.00, step = 0.01)
    play = st.button("Play", type = "primary", use_container_width= True)
    if play:
        mean = [mean1, mean2]
        cov = [[1, 0.5], [0.5, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 1000).T
        
        st.header("Original KDEPlot")
        # Create contour plot
        sns.kdeplot(x = x, y = y, fill = True)

        # Set plot labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("""Bivariate Contour Plot
        $\mu_1 = {}, \mu_2 = {}$""".format(mean[0], mean[1]))

        # Display the plot
        plt.show()
        st.pyplot()
        list = np.array([])
        mybar = st.progress(0, text = "Learning...")
        iterations = 50000
        for iteration in range(iterations):
            a = random.uniform(float(np.min(x)), float(np.max(x)))
            b = random.uniform(float(np.min(y)), float(np.max(y)))
            u = random.uniform(0, 1)
            c = bivariate_normal_pdf(a, b, mean, np.array(cov))
            if iteration == 0:
                list = np.array([[a,b]])
            else:
                if c > u :
                    list = np.concatenate((list, np.array([[a,b]])), axis = 0)
            if iteration < iterations * 0.8:
                # st.write(iteration, iterations)
                mybar.progress(float(iteration)/iterations, text = "Learning...")
            else:
                if iteration < iterations - 1:
                    mybar.progress(float(iteration)/iterations, text = "Almost There...")
                else:
                    mybar.progress(float(iteration)/iterations, text = "Done!")

        st.header("Sampled KDEPlot")
        sns.kdeplot(x = list[:, 0], y = list[:, 1], fill = True, label = "Sampled")
        sns.kdeplot(x = x, y = y, fill = True, label = "True")
        st.pyplot()

        st.header("Linear Regression Results")
        X = np.linspace(-100, 100, 1000)
        Y = np.mean(list[:, 0] * X.reshape(-1,1)  + list[:, 1], axis = 1)
        std = np.mean(list[:, 0] * X.reshape(-1,1) + list[:,1], axis = 1)
        plt.plot(X, Y, label = "True Plot")
        plt.fill_between(X, Y - std, Y + std, color = "gray", alpha = 0.5, label = "uncertainty")
        plt.plot(X, mean1*X + mean2, label = "Predictions")
        plt.legend(loc = "best")
        st.pyplot()

if "Polynomial Regression" in task:
  degree = st.slider("Degree: ", 1, 10, step = 1)
  play = st.button("Play", type = "primary", use_container_width = True)
  if play:
    x = torch.linspace(-3, 3, 100).to(device)
    ground_truth_params = torch.randint(-10, 10, (degree + 1, ))
    ground_truth = torch.tensor([])
    for i in range(ground_truth_params.shape[0]):
      if i == 0:
        ground_truth = ground_truth_params[i] * torch.pow(x, int(ground_truth_params.shape[0]) - 1 - i)
      else:
        ground_truth += ground_truth_params[i] * torch.pow(x, int(ground_truth_params.shape[0]) - 1 - i)
    eps = torch.randn_like(x) * 1.0
    y = ground_truth + eps
    plt.plot(x.cpu(), ground_truth.cpu(), label = "Ground Truth")
    plt.scatter(x.cpu(), y.cpu(), label = "Data")
    plt.legend(loc = "best")
    plt.show()
    st.pyplot()

    st.header("Results using Hamiltorch")
    x0 = torch.randint(-20, 20, (degree + 1, )).type(torch.float)
    num_samples = 1000
    params_hmc = hamiltorch.sample(log_prob_func = log_posterior, params_init = x0, num_samples = num_samples, num_steps_per_sample= 5, step_size = 0.01)
    params_hmc = torch.stack(params_hmc)
    # Plot the traces corresponding to the two parameters
    fig, axes = plt.subplots(x0.shape[0], 1, sharex=True)

    for i, param_vals in enumerate(params_hmc.T):
        axes[i].plot(param_vals, label='Trace')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel(fr'$\theta_{i}$')

    # Plot the true values as well
    for i, param_vals in enumerate(ground_truth_params):
        axes[i].axhline(param_vals, color='C1', label='Ground truth')
        axes[i].legend()
    st.pyplot()

    # Plot the posterior predictive distribution
    plt.figure()
    plt.scatter(x.cpu(), y.cpu(), label='Data', color='C0')
    plt.plot(x.cpu(), ground_truth.cpu(), label='Ground truth', color='C1', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')

    # Get posterior samples. Thin first 100 samples to remove burn-in
    posterior_samples = params_hmc[30:]
    for i in range(posterior_samples.shape[1]):
      if i == 0:
        y_hat = posterior_samples[:, i].unsqueeze(1) * torch.pow(x.cpu(), posterior_samples.shape[1] - 1 - i)
      else:
        y_hat += posterior_samples[:, i].unsqueeze(1) * torch.pow(x.cpu(), posterior_samples.shape[1] - 1 - i)
    # Plot mean and 95% confidence interval

    plt.plot(x.cpu(), y_hat.mean(axis=0), label='Mean', color='C2')
    plt.fill_between(x.cpu(), y_hat.mean(axis=0) - 2 * y_hat.std(axis=0), y_hat.mean(axis=0) + 2 * y_hat.std(axis=0), alpha=0.5, label='95% CI', color='C2')
    plt.legend()
    st.pyplot()

    st.header("Results from Scratch")
    list_ = polynomial_samples_list(x, y, -10, 10, 2)

    fig, axes = plt.subplots(degree + 1, 1, sharex=True)

    for i, param_vals in enumerate(list_.cpu().T):
        axes[i].plot(param_vals, label='Trace')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel(fr'$\theta_{i}$')

    # Plot the true values as well
    for i, param_vals in enumerate(ground_truth_params):
        axes[i].axhline(param_vals, color='C1', label='Ground truth')
        axes[i].legend()
    st.pyplot()

    # Plot the posterior predictive distribution
    plt.figure()
    plt.scatter(x.cpu(), y.cpu(), label='Data', color='C0')
    plt.plot(x.cpu(), ground_truth.cpu(), label='Ground truth', color='C1', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')

    # Get posterior samples. Thin first 100 samples to remove burn-in
    posterior_samples = list_.cpu()[30:]
    for i in range(posterior_samples.shape[1]):
      if i == 0:
        y_hat = posterior_samples[:, i].unsqueeze(1) * torch.pow(x.cpu(), posterior_samples.shape[1] - 1 - i)
      else:
        y_hat += posterior_samples[:, i].unsqueeze(1) * torch.pow(x.cpu(), posterior_samples.shape[1] - 1 - i)
    # Plot mean and 95% confidence interval

    plt.plot(x.cpu(), y_hat.mean(axis=0), label='Mean', color='C2')
    plt.fill_between(x.cpu(), y_hat.mean(axis=0) - 2 * y_hat.std(axis=0), y_hat.mean(axis=0) + 2 * y_hat.std(axis=0), alpha=0.5, label='95% CI', color='C2')
    plt.legend()
    st.pyplot()