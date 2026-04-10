import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange
import random
import warnings
import math
import warnings
from botorch.exceptions.warnings import OptimizationWarning

# Suppress BoTorch Optimization warnings like "ABNORMAL_TERMINATION_IN_LNSRCH"
warnings.filterwarnings("ignore", category=OptimizationWarning)

# Suppress BoTorch input scaling warnings
from botorch.exceptions import InputDataWarning
warnings.filterwarnings("ignore", category=InputDataWarning)

# Suppress numerical warnings from Cholesky decomposition
from linear_operator.utils.warnings import NumericalWarning
warnings.filterwarnings("ignore", category=NumericalWarning)

# Suppress general user warnings (e.g., float32 instead of float64)
warnings.filterwarnings("ignore", category=UserWarning)


def make_grid(dim, den):
    axis = torch.linspace(0, 1, den)
    mesh = torch.meshgrid(*[axis for _ in range(dim)], indexing='ij')
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)

def compute_noise(f,x_dim,c_dim):
    N_sim = 10000
    X_rand = torch.rand(N_sim, x_dim)
    C_rand = torch.rand(N_sim, c_dim)
    f_vals = torch.tensor([f(x, c).item() for x, c in zip(X_rand, C_rand)])
    f_var = torch.var(f_vals)
    noise_var = 0.1 * f_var
    return noise_var.sqrt()
    

def create_perturbed_f_k(f, x_dim, c_dim):
    # Use torch for everything
    a1 = torch.rand(x_dim)                   # scale x ~ Uniform(0,1)
    v = torch.randn(x_dim)                   # shift x ~ Normal(0,1)

    a2 = torch.rand(c_dim)                   # scale c ~ Uniform(0,1)
    u = torch.randn(c_dim)                   # shift c ~ Normal(0,1)

    def f_k(x, ctx):
        x_perturbed = a1 * x + v
        c_perturbed = a2 * ctx + u
        return f(x_perturbed, c_perturbed)

    return f_k

def compute_client_extremes(f_ks, X_space, C_space, mode='max'):
    """Compute the max or min f(x, c) over X_space for each c in C_space, for each client."""
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
    """Compute the normalizing constant as the sum of (max - min) for each context per client."""
    K = len(client_max_y)
    normalizing_const = {
        k: sum(max_v - min_v for max_v, min_v in zip(client_max_y[k], client_min_y[k]))
        for k in range(K)
    }
    return normalizing_const

def run_one_client_step(args):
    (k, f_k, model, X_space, C_space, client_Dx_k, client_Dy_k, client_Dc_k, 
     x_dim, noise_std, client_max_y_k, client_min_y_k, normalizing_const_k) = args

    f_sample, X_repeat, C_rep = sample_posterior_over_grid(model, X_space, C_space)
    
    regret_diffs = compute_regret_diffs_for_client(
        C_space=C_space,
        X_space=X_space,
        X_repeat=X_repeat,
        C_rep=C_rep,
        f_sample=f_sample,
        client_Dx_k=client_Dx_k,
        client_Dy_k=client_Dy_k,
        client_Dc_k=client_Dc_k,
        x_dim=x_dim
    )

    regret_diffs.sort(reverse=True, key=lambda x: x[0].item())
    c_t = regret_diffs[0][1]
    x_star_t = regret_diffs[0][2]
    y_t = f_k(x_star_t, c_t) + noise_std * torch.randn(1)

    # Update best y for normalization
    c_idx = torch.norm(C_space - c_t, dim=1) < 1e-6
    c_idx_int = torch.where(c_idx)[0].item()
    updated_min_y = client_min_y_k[c_idx_int]
    new_y_val = f_k(x_star_t, c_t).item()
    if new_y_val > updated_min_y:
        client_min_y_k[c_idx_int] = new_y_val

    return k, x_star_t, c_t, y_t, client_min_y_k


def compute_regret_diffs_for_client(C_space, X_space, X_repeat, C_rep, f_sample, 
                                     client_Dx_k, client_Dy_k, client_Dc_k, x_dim):

    regret_diffs = []

    for c in C_space:
        c_mask = torch.norm(C_rep - c, dim=1) < 1e-6
        x_candidates = X_repeat[c_mask]
        f_vals_c = f_sample[c_mask]

        if len(f_vals_c) == 0:
            continue  # skip if no data matches context

        x_star_idx = torch.argmax(f_vals_c)
        x_star = x_candidates[x_star_idx]
        f_x_star_val = f_vals_c[x_star_idx]

        # Check if any previous observation exists for this c
        mask = torch.tensor([torch.allclose(c, d_c, atol=1e-5) for d_c in client_Dc_k])

        if mask.any():
            idx = [i for i, m in enumerate(mask) if m]
            dy_vals = torch.tensor([client_Dy_k[i] for i in idx])
            idx_best = idx[torch.argmax(dy_vals)]
            mu_star = client_Dx_k[idx_best]
        else:
            mu_star = X_space[random.randint(0, len(X_space) - 1)]

        dist = ((X_repeat[:, :x_dim] - mu_star) ** 2).sum(dim=1).sqrt() + ((C_rep - c) ** 2).sum(dim=1).sqrt()
        nearest_idx = torch.argmin(dist)
        f_mu_val = f_sample[nearest_idx]

        regret_diffs.append((f_x_star_val - f_mu_val, c, x_star))

    return regret_diffs


def train_gp_surrogate(client_Dx_k, client_Dc_k, client_Dy_k):
    train_x = torch.cat([torch.stack(client_Dx_k), torch.stack(client_Dc_k)], dim=1)
    train_y = torch.stack(client_Dy_k).squeeze(-1)
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def sample_posterior_over_grid(model, X_space, C_space):
    x_len = X_space.shape[0]
    c_len = C_space.shape[0]

    X_repeat = X_space.repeat_interleave(c_len, dim=0)  # (x_len * c_len, x_dim)
    C_rep = C_space.repeat(x_len, 1)                    # (x_len * c_len, c_dim)
    test_points = torch.cat([X_repeat, C_rep], dim=1)   # (x_len * c_len, x_dim + c_dim)

    posterior = model.posterior(test_points)
    f_sample = posterior.sample(sample_shape=torch.Size([1])).squeeze(0).squeeze(-1)

    return f_sample, X_repeat, C_rep


def FMTS_indep(f_ks, K = 1, T=100, t_0=5, x_dim=1, c_dim=1, c_dense=10, x_dense=10):
    noise_std = compute_noise(f_ks[1],x_dim,c_dim)

    X_space = make_grid(x_dim, x_dense)
    C_space = make_grid(c_dim, c_dense)
    
    # store the historical observations
    client_Dx = {k: [] for k in range(K)} 
    client_Dc = {k: [] for k in range(K)} 
    client_Dy = {k: [] for k in range(K)} 

    # --- Initial random observations --- record the index for (x,c)
    for _ in range(t_0):
        for k in range(K):
            x = X_space[torch.randint(len(X_space), (1,))][0]
            c = C_space[torch.randint(len(C_space), (1,))][0]
            y = f_ks[k](x, c) + noise_std * torch.randn(1)
            client_Dx[k].append(x), client_Dc[k].append(c), client_Dy[k].append(y)
            
    # extreme values used for defining regret
    client_max_y = compute_client_extremes(f_ks, X_space, C_space, mode='max')
    client_min_y = compute_client_extremes(f_ks, X_space, C_space, mode='min')
    normalizing_const = compute_normalizing_const(client_max_y, client_min_y)

    
    regret = {k: [] for k in range(K)}
    # --- Main Thompson sub-optimal loop ---
    for t in trange(t_0 + 1, T + 1, desc="Running MTSS indep"):
  
        for k in range(K):
            # compute the regret
            regret[k].append(sum([max_v - min_v for max_v, min_v in zip(client_max_y[k], client_min_y[k])])/normalizing_const[k])
            model = train_gp_surrogate(client_Dx[k], client_Dc[k], client_Dy[k]) # training surrogate
            f_sample, X_repeat, C_rep = sample_posterior_over_grid(model, X_space, C_space) # sample from the posterior

            regret_diffs = compute_regret_diffs_for_client(
                C_space=C_space,
                X_space=X_space,
                X_repeat=X_repeat,
                C_rep=C_rep,
                f_sample=f_sample,
                client_Dx_k=client_Dx[k],
                client_Dy_k=client_Dy[k],
                client_Dc_k=client_Dc[k],
                x_dim=x_dim
            )

            regret_diffs.sort(reverse=True, key=lambda x: x[0].item())
            c_t = regret_diffs[0][1]  # stage 1: select c first
            x_star_t = regret_diffs[0][2]  # stage 2: select x according to the sample


            y_t = f_ks[k](x_star_t, c_t) + noise_std * torch.randn(1)
            client_Dx[k].append(x_star_t), client_Dc[k].append(c_t), client_Dy[k].append(y_t)

            # update the index of current best
            c_idx = torch.norm(C_space - c_t, dim=1) < 1e-6
            c_idx_int = torch.where(c_idx)[0].item()  # Get the actual integer index
            if f_ks[k](x_star_t, c_t).item() > client_min_y[k][c_idx_int]:
                client_min_y[k][c_idx_int] = f_ks[k](x_star_t, c_t).item()

    return client_Dx, client_Dc, client_Dy, regret, [sum(regret[k][t] for k in range(K)) for t in range(T - t_0)]


def FMTS_rand(f_ks, K = 1, T=100, t_0=5, x_dim=1, c_dim=1, c_dense=10, x_dense=10):
    noise_std = compute_noise(f_ks[1],x_dim,c_dim)

    X_space = make_grid(x_dim, x_dense)
    C_space = make_grid(c_dim, c_dense)
    
    # store the historical observations
    client_Dx = {k: [] for k in range(K)} 
    client_Dc = {k: [] for k in range(K)} 
    client_Dy = {k: [] for k in range(K)} 

    # --- Initial random observations --- record the index for (x,c)
    for _ in range(t_0):
        for k in range(K):
            x = X_space[torch.randint(len(X_space), (1,))][0]
            c = C_space[torch.randint(len(C_space), (1,))][0]
            y = f_ks[k](x, c) + noise_std * torch.randn(1)
            client_Dx[k].append(x), client_Dc[k].append(c), client_Dy[k].append(y)
            
    # extreme values used for defining regret
    client_max_y = compute_client_extremes(f_ks, X_space, C_space, mode='max')
    client_min_y = compute_client_extremes(f_ks, X_space, C_space, mode='min')
    normalizing_const = compute_normalizing_const(client_max_y, client_min_y)

    
    regret = {k: [] for k in range(K)}
    # --- Main Thompson sub-optimal loop ---
    for t in trange(t_0 + 1, T + 1, desc="Running MTSS"):
  
        for k in range(K):
            regret[k].append(sum([max_v - min_v for max_v, min_v in zip(client_max_y[k], client_min_y[k])])/normalizing_const[k])
            x_star_t = X_space[torch.randint(len(X_space), (1,))][0]  # stage 1: select c first
            c_t = C_space[torch.randint(len(C_space), (1,))][0]  # stage 2: select x according to the sample


            y_t = f_ks[k](x_star_t, c_t) + noise_std * torch.randn(1)
            client_Dx[k].append(x_star_t), client_Dc[k].append(c_t), client_Dy[k].append(y_t)

            # update the index of current best
            c_idx = torch.norm(C_space - c_t, dim=1) < 1e-6
            c_idx_int = torch.where(c_idx)[0].item()  # Get the actual integer index
            if f_ks[k](x_star_t, c_t).item() > client_min_y[k][c_idx_int]:
                client_min_y[k][c_idx_int] = f_ks[k](x_star_t, c_t).item()

    return client_Dx, client_Dc, client_Dy, regret, [sum(regret[k][t] for k in range(K)) for t in range(T - t_0)]


def FMTS_fd(f_ks, K = 1, T=100, t_0=5, x_dim=1, c_dim=1, c_dense=10, x_dense=10):
    noise_std = compute_noise(f_ks[1],x_dim,c_dim)

    X_space = make_grid(x_dim, x_dense)
    C_space = make_grid(c_dim, c_dense)
    
    # store the historical observations
    client_Dx = {k: [] for k in range(K)} 
    client_Dc = {k: [] for k in range(K)} 
    client_Dy = {k: [] for k in range(K)} 

    # --- Initial random observations --- record the index for (x,c)
    for _ in range(t_0):
        for k in range(K):
            x = X_space[torch.randint(len(X_space), (1,))][0]
            c = C_space[torch.randint(len(C_space), (1,))][0]
            y = f_ks[k](x, c) + noise_std * torch.randn(1)
            client_Dx[k].append(x), client_Dc[k].append(c), client_Dy[k].append(y)
            
    # extreme values used for defining regret
    client_max_y = compute_client_extremes(f_ks, X_space, C_space, mode='max')
    client_min_y = compute_client_extremes(f_ks, X_space, C_space, mode='min')
    normalizing_const = compute_normalizing_const(client_max_y, client_min_y)

    
    regret = {k: [] for k in range(K)}
    # --- Main Thompson sub-optimal loop ---
    for t in trange(t_0 + 1, T + 1, desc="Running MTSS fed"):

        proposed_x = []
        proposed_c = []
        
        for k in range(K):
            # compute the regret
            regret[k].append(sum([max_v - min_v for max_v, min_v in zip(client_max_y[k], client_min_y[k])])/normalizing_const[k])
            model = train_gp_surrogate(client_Dx[k], client_Dc[k], client_Dy[k]) # training surrogate
            f_sample, X_repeat, C_rep = sample_posterior_over_grid(model, X_space, C_space) # sample from the posterior

            regret_diffs = compute_regret_diffs_for_client(
                C_space=C_space,
                X_space=X_space,
                X_repeat=X_repeat,
                C_rep=C_rep,
                f_sample=f_sample,
                client_Dx_k=client_Dx[k],
                client_Dy_k=client_Dy[k],
                client_Dc_k=client_Dc[k],
                x_dim=x_dim
            )

            regret_diffs.sort(reverse=True, key=lambda x: x[0].item())
            c_t = regret_diffs[0][1]  # stage 1: select c first
            x_star_t = regret_diffs[0][2]  # stage 2: select x according to the sample
            proposed_x.append(x_star_t)
            proposed_c.append(c_t)

            # --- Count how many other clients have observed this context ---
        
        for k in range(K):
            count_list = []
            for j in range(K):
                count = sum(torch.allclose(proposed_c[k], c_j, atol=1e-5) for c_j in client_Dc[j])
                count_list.append(count)
            
            if sum(count_list) == 0:
                cand = None
            else:
                max_obs = max(count_list)
                other_clients = [j for j in range(K)]
                candidate_clients = [j for i, j in enumerate(other_clients) if count_list[i] == max_obs]
                cand = random.choice(candidate_clients) if candidate_clients else None
                
            z = torch.rand(1)
            if cand is None or z <1/2**(t-t_0+1) or cand == k:
                x_star_t = proposed_x[k]
                
            else:           
                match_indices = [i for i, c_j in enumerate(client_Dc[cand]) if torch.allclose(proposed_c[k], c_j, atol=1e-5)]
                y_vals = [client_Dy[cand][i] for i in match_indices]
                best_idx = match_indices[torch.argmax(torch.tensor(y_vals))]
                x_star_t = client_Dx[cand][best_idx]
                
            y_t = f_ks[k](x_star_t, proposed_c[k]) + noise_std * torch.randn(1)
            client_Dx[k].append(x_star_t), client_Dc[k].append(proposed_c[k]), client_Dy[k].append(y_t)

            # update the index of current best
            c_idx = torch.norm(C_space - c_t, dim=1) < 1e-6
            c_idx_int = torch.where(c_idx)[0].item()  # Get the actual integer index
            if f_ks[k](x_star_t, c_t).item() > client_min_y[k][c_idx_int]:
                client_min_y[k][c_idx_int] = f_ks[k](x_star_t, c_t).item()

    return client_Dx, client_Dc, client_Dy, regret, [sum(regret[k][t] for k in range(K)) for t in range(T - t_0)]






def levy_1x1(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Levy function with 1D design x and 1D context c → treated as 2D input.
    """
    z = torch.cat([x, c])  # shape (2,)
    w = 1 + (z - 1) / 4

    term1 = torch.sin(math.pi * w[0]) ** 2
    term2 = torch.sum((w[:-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[-1]) ** 2)

    return -term1 - term2 - term3

import torch
import math
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim import optimize_acqf

# ----- 1. Simulate data -----
x_dim, c_dim = 2, 2
n = 20
X = torch.rand(n, x_dim)
C = torch.rand(n, c_dim)
XC = torch.cat([X, C], dim=1)

def levy_1x1(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    z = torch.cat([x, c], dim=-1)
    w = 1 + (z - 1) / 4
    term1 = torch.sin(math.pi * w[..., 0]) ** 2
    term2 = torch.sum((w[..., :-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[..., :-1] + 1) ** 2), dim=-1)
    term3 = (w[..., -1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[..., -1]) ** 2)
    return -(term1 + term2 + term3)

y = levy_1x1(X, C).unsqueeze(-1)

