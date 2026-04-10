import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange
import numpy as np
import math
from helper_func import *
from botorch.acquisition.analytic import PosteriorMean
from botorch.models.transforms import Normalize, Standardize
import random
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples


def collect_good_idx(des_idx, des):
    idx_tensor = torch.stack(des_idx).T
    num_experts = idx_tensor.shape[1]

    selected_designs = torch.stack([des[idx_tensor[:, i]] for i in range(num_experts)], dim=1)
    rand_idx = torch.randint(0, num_experts, (selected_designs.shape[0],))
    rand_designs = selected_designs[torch.arange(selected_designs.shape[0]), rand_idx]

    # Compute distances to all candidates and pick closest index
    dists = torch.cdist(rand_designs, des)
    closest_idxs = torch.argmin(dists, dim=1)  # [num_contexts]
    return closest_idxs

def sum_variance_per_client_gp_from_separate_dicts(
    client_Dx,
    client_Dc,
    client_Dy,
    c_t,
    x_cand,
    seed=42
):
    torch.manual_seed(seed)
    random.seed(seed)

    c_t = c_t.unsqueeze(0)  # shape (1, c_dim)
    client_id = random.choice(list(client_Dx.keys()))

    x_list = client_Dx[client_id]
    c_list = client_Dc[client_id]
    y_list = client_Dy[client_id]

    x_dim = x_list[0].numel()
    c_dim = c_list[0].numel()

    X = torch.cat([torch.stack(x_list), torch.stack(c_list)], dim=1)
    Y = torch.stack(y_list).float().reshape(-1, 1)

    # Train GP
    model = SingleTaskGP(
        X, Y,
        input_transform=Normalize(X.shape[-1]),
        outcome_transform=Standardize(1)
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Create (x, c_t) combinations
    c_repeat = c_t.expand(x_cand.shape[0], -1)  # (N, c_dim)
    xc_eval = torch.cat([x_cand, c_repeat], dim=1)  # (N, x_dim + c_dim)

    # Thompson Sampling over the discrete candidates
    model.eval()
    with torch.no_grad():
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=seed)
        posterior = model.posterior(xc_eval)
        samples = sampler(posterior).squeeze(0)  # (N,)
        best_idx = torch.argmax(samples)
        x_star = x_cand[best_idx]

    return x_star

def optimize_posterior_sample(model, x_mu: torch.Tensor, c: torch.Tensor, x_candidates, seed: int = 20250717):
    # Generate 1000 candidate x values in [0, 1]^x_dim

    # Expand context c to match x_candidates
    c_expanded = c.expand(x_candidates.shape[0], -1)
    xc_input = torch.cat([x_candidates, c_expanded], dim=1)  # shape (1000, x_dim + c_dim)

    # Sample the GP at these 1000 points
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=seed)
    with torch.no_grad():
        posterior = model.posterior(xc_input)
        f_samples = sampler(posterior).squeeze(0)  # shape (1000,)

        # Evaluate also at x_mu
        x_mu_input = torch.cat([x_mu, c.unsqueeze(0)], dim=-1)
        posterior_mu = model.posterior(x_mu_input)
        f_mu_sample = sampler(posterior_mu).squeeze().item()

    # Find the best x according to the sampled value
    best_idx = torch.argmax(f_samples)
    x_star = x_candidates[best_idx].unsqueeze(0)

    return x_star, f_samples[best_idx].item() - f_mu_sample

def optimize_max_across_contexts(model, x_mu: torch.Tensor, x_candidates: torch.Tensor, c_list: torch.Tensor, seed: int = 20250717):

    torch.manual_seed(seed)
    x_mu = x_mu.unsqueeze(1) 
    N, x_dim = x_candidates.shape
    T, c_dim = c_list.shape
    

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=seed)

    f_samples = []
    f_mean = []

    f_mu_sample = []
    for t in range(100):
        c_t = c_list[t]
        # Expand context to match number of candidates
        c_expanded = c_t.expand(N, -1)                # shape (N, c_dim)
        xc = torch.cat([x_candidates, c_expanded], dim=1)  # shape (N, x_dim + c_dim)
        posterior = model.posterior(xc)
        sample = sampler(posterior).squeeze(0)  # (N,)
        mean = posterior.mean.squeeze(-1)  # shape: (T,)

        f_samples.append(sample)
        f_mean.append(mean)
        
    if x_mu.ndim == 3:
        x_mu = x_mu.squeeze(1)
    xmu_ct = torch.cat([x_mu, c_list], dim=1)   # shape (T, x_dim + c_dim)
    posterior_mu = model.posterior(xmu_ct)
    f_mu_sample = sampler(posterior_mu).squeeze(0)  # shape: (T,)


    # Stack into shape (T, N)
    f_samples = torch.stack(f_samples).squeeze(-1)                # shape (T, N)
    f_mean = torch.stack(f_mean).squeeze(-1)                # shape (T, N)

    best_idx = torch.argmax(f_samples, dim=1)                 # (T,)
    best_vals = f_samples[torch.arange(T), best_idx].squeeze()          # (T,)
    improvements = best_vals - f_mu_sample.squeeze(-1) 
    # (T,)
    # Find best improvement over all t
    best_t = torch.argmax(improvements) # c_idx
    x_star_best = x_candidates[best_idx[best_t]] # (1, x_dim)
    c_star_best = c_list[best_t]             # (1, c_dim)

    return x_star_best, c_star_best, best_t, f_samples[best_idx, torch.arange(N)].squeeze(), best_idx, f_mean 


def optimize_posterior_mean(model, c, x_candidates, seed = 20250717):
    # Step 2: Repeat context c to match candidate count
    c_expanded = c.expand(x_candidates.shape[0], -1)
    # Step 3: Form joint input (x, c)
    xc_input = torch.cat([x_candidates, c_expanded], dim=1)
    # Step 4: Predict posterior mean
    with torch.no_grad():
        posterior = model.posterior(xc_input)
        mean_vals = posterior.mean.squeeze(-1)  # shape (1000,)

    # Step 5: Select the best x
    best_idx = torch.argmax(mean_vals)
    x_star = x_candidates[best_idx].unsqueeze(0)
    max_mean = mean_vals[best_idx].item()

    return x_star, max_mean

def compute_current_regret(model, f_k, client_max_y_k, normalizing_const_k, x_cand, c_cand):

    posterior_mean = PosteriorMean(model)
    current_best = []

    for c in c_cand:
        c = c.unsqueeze(0)  # (1, c_dim)
        c_expand = c.expand(250, -1)  # (n_samples, c_dim)
        xc = torch.cat([x_cand, c_expand], dim=1)  # (n_samples, x_dim + c_dim)

        # Evaluate posterior mean
        with torch.no_grad():
            mean = posterior_mean(xc.unsqueeze(1)).squeeze(-1).squeeze(-1)  # shape: (n_samples,)
        
        best_idx = torch.argmax(mean)
        x_star_c = x_cand[best_idx]  # shape: (1, x_dim)
        f_val = f_k(x_star_c, c.squeeze(0)).item()
        current_best.append(f_val)

    regret_value = sum(max_v - curr for max_v, curr in zip(client_max_y_k, current_best)) / normalizing_const_k

    return regret_value


def make_grid(dim, den):
    axis = torch.linspace(0, 1, den)
    mesh = torch.meshgrid(*[axis for _ in range(dim)], indexing='ij')
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)

def compute_noise(f,x_dim,c_dim):
    N_sim = 1000
    X_rand = torch.rand(N_sim, x_dim)
    C_rand = torch.rand(N_sim, c_dim)
    f_vals = torch.tensor([f(x, c).item() for x, c in zip(X_rand, C_rand)])
    f_var = torch.var(f_vals)
    noise_var = 0.1 * f_var
    return noise_var.sqrt()
    
def create_perturbed_f_k(f, x_dim, c_dim):
    # Use torch for everything
    a1 = torch.rand(1)/20                   # scale x ~ Uniform(0,1)
    v = torch.randn(x_dim)/20                   # shift x ~ Normal(0,1)

    a2 = torch.rand(1)/20                   # scale c ~ Uniform(0,1)
    u = torch.randn(c_dim)/20                   # shift c ~ Normal(0,1)

    def f_k(x, ctx):
        x_perturbed = x + v
        c_perturbed = ctx + u
        return f(x_perturbed, c_perturbed)

    return f_k

def finite_client_extremes(f_ks, X_space, C_space, mode='max'):
    K = len(f_ks)
    client_y = {k: [] for k in range(K)}
    for k in range(K):
        f_k = f_ks[k]
        for c in C_space:
            values = [f_k(x, c).item() for x in X_space]
            extreme_val = max(values) if mode == 'max' else min(values)
            client_y[k].append(extreme_val)
    return client_y

def compute_normalizing_const(client_max_y, client_min_y):
    K = len(client_max_y)
    normalizing_const = {
        k: sum(max_v - min_v for max_v, min_v in zip(client_max_y[k], client_min_y[k]))
        for k in range(K)
    }
    return normalizing_const


def train_gp_surrogate(x_list, c_list, y_list):
    x_tensor = torch.stack(x_list)  # (n, 1)
    c_tensor = torch.stack(c_list)  # (n, 1)
    X = torch.cat([x_tensor, c_tensor], dim=1)  # (n, 2)
    y_tensor = torch.stack(y_list).float().reshape(-1, 1)  # (n, 1)
    model = SingleTaskGP(X, y_tensor, input_transform=Normalize(X.size(1)), outcome_transform=Standardize(1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

##########
# Below implement the simulation functions
##########

def levy(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

    z = torch.cat([x, c], dim=-1)               # (..., d)
    z_scaled = 20.0 * z - 10.0                  # rescale to [-10, 10]
    w = 1 + (z_scaled - 1) / 4

    term1 = torch.sin(math.pi * w[..., 0]) ** 2
    term2 = torch.sum((w[..., :-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[..., :-1] + 1) ** 2), dim=-1)
    term3 = (w[..., -1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[..., -1]) ** 2)

    return -(term1 + term2 + term3)  # negative for BO maximization



def branin(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # x: (..., 1)
    # c: (..., 1)
    z = torch.cat([x, c], dim=-1)  # (..., 2)

    # Rescale z from [0, 1]^2 to the Branin domain:
    # x1 ∈ [-5, 10], x2 ∈ [0, 15]
    x1 = z[..., 0] * 15.0 - 5.0
    x2 = z[..., 1] * 15.0

    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c_const = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * math.pi)

    f = a * (x2 - b * x1 ** 2 + c_const * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
    return -f  # for maximization


def ackley(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    z_unit = torch.cat([x, c], dim=-1)  # shape (..., d)
    z = 10.0 * z_unit - 5.0  # same shape as z_unit

    d = z.shape[-1]
    a = 20
    b = 0.2
    c_const = 2 * math.pi

    term1 = -a * torch.exp(-b * torch.norm(z, dim=-1) / math.sqrt(d))
    term2 = -torch.exp(torch.mean(torch.cos(c_const * z), dim=-1))
    result = term1 + term2 + a + math.e

    return -result  # negate for maximization

def hartmann_2x2(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([
        [10.0, 3.0, 17.0, 3.5],
        [0.05, 10.0, 17.0, 0.1],
        [3.0, 3.5, 1.7, 10.0],
        [17.0, 8.0, 0.05, 10.0]
    ])
    P = torch.tensor([
        [0.1312, 0.1696, 0.5569, 0.0124],
        [0.2329, 0.4135, 0.8307, 0.3736],
        [0.2348, 0.1451, 0.3522, 0.2883],
        [0.4047, 0.8828, 0.8732, 0.5743]
    ])

    z = torch.cat([x, c])  # shape (4,)
    inner = torch.sum(A * (z - P) ** 2, dim=1)  # shape (4,)
    result = -torch.sum(alpha * torch.exp(-inner))

    return result

import torch
import pandas as pd
import os

def export_experiment_results(results, label, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def save_dict_to_excel(data_dict, filepath, var_label):
        with pd.ExcelWriter(filepath) as writer:
            for client_id, data_list in data_dict.items():
                df = pd.DataFrame(torch.stack(data_list).numpy())
                df.to_excel(writer, sheet_name=f"{var_label}_client{client_id}", index=False)

    for i, (dx, dc, dy, regret, regret_trace) in enumerate(results):
        # Save dictionaries to Excel
        save_dict_to_excel(dx, os.path.join(output_dir, f"{label}_dx_{i}.xlsx"), "x")
        save_dict_to_excel(dc, os.path.join(output_dir, f"{label}_dc_{i}.xlsx"), "c")
        save_dict_to_excel(dy, os.path.join(output_dir, f"{label}_dy_{i}.xlsx"), "y")

        # Save regret and regret trace to CSV
        df_regret = pd.DataFrame(regret)
        df_regret_trace = pd.DataFrame(torch.tensor(regret_trace).numpy())
        df_regret.to_csv(os.path.join(output_dir, f"{label}_regret_{i}.csv"), index=False)
        df_regret_trace.to_csv(os.path.join(output_dir, f"{label}_regret_trace_{i}.csv"), index=False)


def optimize_max_across_context_fd(model, x_mu: torch.Tensor, x_candidates: torch.Tensor, c_list: torch.Tensor, seed: int = 20250717):

    torch.manual_seed(seed)
    x_mu = x_mu.unsqueeze(1) 
    N, x_dim = x_candidates.shape
    T, c_dim = c_list.shape
    

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=seed)

    f_samples = []
    f_mean = []

    f_mu_sample = []
    for t in range(100):
        c_t = c_list[t]
        # Expand context to match number of candidates
        c_expanded = c_t.expand(N, -1)                # shape (N, c_dim)
        xc = torch.cat([x_candidates, c_expanded], dim=1)  # shape (N, x_dim + c_dim)
        posterior = model.posterior(xc)
        sample = sampler(posterior).squeeze(0)  # (N,)
        mean = posterior.mean.squeeze(-1)  # shape: (T,)
        f_samples.append(sample)
        f_mean.append(mean)
        
    if x_mu.ndim == 3:
        x_mu = x_mu.squeeze(1)
    xmu_ct = torch.cat([x_mu, c_list], dim=1)   # shape (T, x_dim + c_dim)
    posterior_mu = model.posterior(xmu_ct)
    f_mu_sample = sampler(posterior_mu).squeeze(0)  # shape: (T,)


    # Stack into shape (T, N)
    f_samples = torch.stack(f_samples).squeeze(-1)                # shape (T, N)
    f_mean = torch.stack(f_mean).squeeze(-1)                # shape (T, N)

    best_idx = torch.argmax(f_samples, dim=1)                 # (T,)
    best_vals = f_samples[torch.arange(T), best_idx].squeeze()          # (T,)
    improvements = best_vals - f_mu_sample.squeeze(-1) 
    # (T,)
    # Find best improvement over all t
    best_t = torch.argmax(improvements) # c_idx
    x_star_best = x_candidates[best_idx[best_t]] # (1, x_dim)
    c_star_best = c_list[best_t]             # (1, c_dim)

    return x_star_best, c_star_best, [x_candidates[i] for i in best_idx], best_vals, best_t, f_mean


def optimize_max_across_context_fts(model, x_mu: torch.Tensor, x_candidates: torch.Tensor, c_list: torch.Tensor, seed: int = 20250717):

    torch.manual_seed(seed)
    x_mu = x_mu.unsqueeze(1) 
    N, x_dim = x_candidates.shape
    T, c_dim = c_list.shape
    

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=seed)

    f_samples = []
    f_mean = []

    f_mu_sample = []
    for t in range(100):
        c_t = c_list[t]
        # Expand context to match number of candidates
        c_expanded = c_t.expand(N, -1)                # shape (N, c_dim)
        xc = torch.cat([x_candidates, c_expanded], dim=1)  # shape (N, x_dim + c_dim)
        posterior = model.posterior(xc)
        sample = sampler(posterior).squeeze(0)  # (N,)
        mean = posterior.mean.squeeze(-1)  # shape: (T,)
        f_samples.append(sample)
        f_mean.append(mean)
        
    if x_mu.ndim == 3:
        x_mu = x_mu.squeeze(1)
    xmu_ct = torch.cat([x_mu, c_list], dim=1)   # shape (T, x_dim + c_dim)
    posterior_mu = model.posterior(xmu_ct)
    f_mu_sample = sampler(posterior_mu).squeeze(0)  # shape: (T,)


    # Stack into shape (T, N)
    f_samples = torch.stack(f_samples).squeeze(-1)                # shape (T, N)
    f_mean = torch.stack(f_mean).squeeze(-1)                # shape (T, N)

    best_idx = torch.argmax(f_samples, dim=1)                 # (T,)
    best_vals = f_samples[torch.arange(T), best_idx].squeeze()          # (T,)
    improvements = best_vals - f_mu_sample.squeeze(-1) 
    # (T,)
    # Find best improvement over all t
    best_t = torch.argmax(improvements) # c_idx
    x_star_best = x_candidates[best_idx[best_t]] # (1, x_dim)
    c_star_best = c_list[best_t]             # (1, c_dim)

    return x_star_best, c_star_best, [x_candidates[i] for i in best_idx], best_vals, best_t, f_samples

def considered_optimal_regret_seq(model, f_k, client_max_y_k, normalizing_const_k, x_cand, c_cand):
    posterior_mean = PosteriorMean(model)
    current_best = []

    for c in c_cand:
        c = c.unsqueeze(0)  # (1, c_dim)
        c_expand = c.expand(250, -1)  # (n_samples, c_dim)
        xc = torch.cat([x_cand, c_expand], dim=1)  # (n_samples, x_dim + c_dim)

        # Evaluate posterior mean
        with torch.no_grad():
            mean = posterior_mean(xc.unsqueeze(1)).squeeze(-1).squeeze(-1)  # shape: (n_samples,)
        
        best_idx = torch.argmax(mean)
        x_star_c = x_cand[best_idx]  # shape: (1, x_dim)
        f_val = f_k(x_star_c, c.squeeze(0)).item()
        current_best.append(f_val)

    # Compute raw gaps
    gaps = [max_v - curr for max_v, curr in zip(client_max_y_k, current_best)]

    # Total regret normalized
    regret_value = sum(gaps) / normalizing_const_k

    # Build decreasing list
    sorted_gaps = sorted(gaps, reverse=True)
    total = sum(sorted_gaps)
    decrease_list = [total]
    for g in sorted_gaps:
        total -= g
        decrease_list.append(total)
    # Normalize if needed
    decrease_list = [val / normalizing_const_k for val in decrease_list]

    return decrease_list


# utility function for rff approxiimation

def create_shared_rff_basis(input_dim, num_features=1000, lengthscale=1.0, outputscale=1.0, seed=42):
    """Create and save a shared RFF basis (W, b)."""
    torch.manual_seed(seed)
    W = torch.randn(num_features, input_dim) / lengthscale
    b = 2 * torch.pi * torch.rand(num_features)
    torch.save({'W': W, 'b': b, 'lengthscale': lengthscale, 'outputscale': outputscale}, "shared_rff_basis.pt")
    return W, b

def compute_rff_weights_from_gp(model, W, b):
    """Compute RFF posterior weights (w_hat) from a trained BoTorch GP model."""
    X = model.train_inputs[0]
    y = model.train_targets.unsqueeze(-1)
    kernel = model.covar_module.base_kernel
    lengthscale = kernel.lengthscale.detach().squeeze()
    outputscale = model.covar_module.outputscale.item()
    noise = model.likelihood.noise.item()
    num_features = W.shape[0]
    
    def phi(X):
        proj = X @ W.T + b
        return (outputscale**0.5) * (2.0 / num_features)**0.5 * torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)
    
    Phi = phi(X)
    A = Phi.T @ Phi + noise * torch.eye(Phi.shape[1])
    w_hat = torch.linalg.solve(A, Phi.T @ y)
    return w_hat, lengthscale, outputscale, noise

def optimize_max_across_context_fd_rff(W, b, w_hat, lengthscale, outputscale, x_mu, x_candidates, c_list, seed=20250717):

    torch.manual_seed(seed)
    x_mu = x_mu.unsqueeze(1)
    N, x_dim = x_candidates.shape
    T, c_dim = c_list.shape
    num_features = W.shape[0]
    
    def phi(X):
        proj = X @ W.T + b
        return (outputscale**0.5) * (2.0 / num_features)**0.5 * torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)
    
    f_mean = []
    for t in range(T):
        c_t = c_list[t]
        c_expanded = c_t.expand(N, -1)
        xc = torch.cat([x_candidates, c_expanded], dim=1)
        Phi_t = phi(xc)
        mean = (Phi_t @ w_hat).squeeze(-1)
        f_mean.append(mean)
    
    f_mean = torch.stack(f_mean)  # shape (T, N)
    
    # Compute baseline means for each context (x_mu, c_t)
    if x_mu.ndim == 3:
        x_mu = x_mu.squeeze(1)
    if x_mu.shape[0] == 1 and c_list.shape[0] > 1:
        x_mu = x_mu.expand(c_list.shape[0], -1)
    xmu_ct = torch.cat([x_mu, c_list], dim=1)
    Phi_mu = phi(xmu_ct)
    f_mu_sample = (Phi_mu @ w_hat).squeeze(-1)
    
    # Compute improvements
    best_idx = torch.argmax(f_mean, dim=1)
    best_vals = f_mean[torch.arange(T), best_idx].squeeze()
    improvements = best_vals - f_mu_sample.squeeze(-1)
    best_t = torch.argmax(improvements)
    
    x_star_best = x_candidates[best_idx[best_t]]
    c_star_best = c_list[best_t]
    
    return x_star_best, c_star_best, [x_candidates[i] for i in best_idx], best_vals, best_t, f_mean
