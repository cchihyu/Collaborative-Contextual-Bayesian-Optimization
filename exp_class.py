import torch
import numpy as np
from tqdm import trange
import warnings
from multiprocessing import Pool, cpu_count
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import math
from botorch.acquisition.analytic import PosteriorMean
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling import SobolQMCNormalSampler
from helper_CMTS import *
import pandas as pd
import os
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

def model_train_and_acquire_fd(arg):
    k, Dx, Dc, Dy, x_dim, c_dim, seed = arg
    model = train_gp_surrogate(Dx, Dc, Dy)

    c_consider = torch.rand(250, c_dim)
    x_consider = torch.rand(250, x_dim)
    best_x_mu = [optimize_posterior_mean(model, c, x_consider)[0] for c in c_consider]
    best_x_mu = torch.stack([x.squeeze() for x in best_x_mu])

    x_t, c_t = optimize_max_across_contexts(model, best_x_mu, x_consider, c_consider, seed=seed)
    return k, model, x_t, c_t, x_consider

def eval_decision_and_forward(arg):
    c_t, proposed_x_k, x_consider, Dx, Dc, Dy, seed, t_curr = arg
    if torch.rand(1).item() < 1 / (t_curr):
        x_t = sum_variance_per_client_gp_from_separate_dicts(
            Dx, Dc, Dy, c_t, x_consider, seed
        )
    else:
        x_t = proposed_x_k
    return  x_t, c_t

def evaluate_c(args):
    model, c, x_dim = args
    x_mu, _ = optimize_posterior_mean(model, c, x_dim)
    x_star, val = optimize_posterior_sample(model, x_mu, c, x_dim)
    return val.item(), c, x_star.squeeze(0)

# main experiment function
class FederatedThompsonSampler:
    def __init__(self, f, x_dim, c_dim, K, T, t_0, f_ks, seed=42):
        self.f_ks = f_ks
        self.K = K
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.T = T
        self.t_0 = t_0
        self.seed = seed
        self.c_cal = torch.rand(250, self.c_dim)
        self.x_cal = torch.rand(250, self.x_dim)

        self.noise_std = compute_noise(self.f_ks[0], x_dim, c_dim)
        self.client_max_y = finite_client_extremes(self.f_ks, self.x_cal, self.c_cal, mode='max')
        self.client_min_y = finite_client_extremes(self.f_ks, self.x_cal, self.c_cal, mode='min')
        self.normalizing_const = compute_normalizing_const(self.client_max_y, self.client_min_y)

        self.client_Dx = {k: [] for k in range(self.K)}
        self.client_Dc = {k: [] for k in range(self.K)}
        self.client_Dy = {k: [] for k in range(self.K)}
        self.regret = {k: [] for k in range(self.K)}
        T = 10  # or any number of contexts you want
        c_dim = self.c_dim
        x_dim = self.x_dim

        self.C = [torch.rand(100, c_dim) for _ in range(self.T+self.t_0)] 
        self.X = [torch.rand(100, x_dim) for _ in range(self.T+self.t_0)] 
        # Initial random observations
        for _ in range(t_0):
            c_cand = self.C[_]
            x_cand = self.X[_]
            for k in range(self.K):
                c = c_cand[torch.randint(0, c_cand.shape[0], (1,)).item()]
                x = x_cand[torch.randint(0, x_cand.shape[0], (1,)).item()]
                y = self.f_ks[k](x, c) + self.noise_std * torch.randn(1)
                self.client_Dx[k].append(x)
                self.client_Dc[k].append(c)
                self.client_Dy[k].append(y)

        # Store initial data for reuse
        self._store_initial_data()

    def _store_initial_data(self):
        self.init_Dx = {k: list(self.client_Dx[k]) for k in range(self.K)}
        self.init_Dc = {k: list(self.client_Dc[k]) for k in range(self.K)}
        self.init_Dy = {k: list(self.client_Dy[k]) for k in range(self.K)}

    def reset_data(self):
        self.client_Dx = {k: list(self.init_Dx[k]) for k in range(self.K)}
        self.client_Dc = {k: list(self.init_Dc[k]) for k in range(self.K)}
        self.client_Dy = {k: list(self.init_Dy[k]) for k in range(self.K)}
        self.regret = {k: [] for k in range(self.K)}

    def _step(self, k, model):
        return compute_current_regret(
            model,
            self.f_ks[k],
            self.client_max_y[k],
            self.normalizing_const[k],
            self.x_cal,
            self.c_cal
        )

    # compute the context-specific optimal design
    def run_optimal(self):
        optimal_reg = []
        for k in range(self.K):
            model = train_gp_surrogate(self.client_Dx[k], self.client_Dc[k], self.client_Dy[k])
            optimal_reg.append(considered_optimal_regret_seq(
            model,             
            self.f_ks[k],
            self.client_max_y[k],
            self.normalizing_const[k],
            self.x_cal,
            self.c_cal))
        return [sum(seq[j] for seq in optimal_reg if len(seq) > j) / sum(len(seq) > j for seq in optimal_reg) for j in range(20*(self.c_dim+self.x_dim))]

    # run independent MTS (offline contextual Bayesian optimization)
    def run_indep(self):
        for t in trange(self.t_0 + 1, self.T + 1, desc="Running FMTS_indep"):
            for k in range(self.K):    

                model = train_gp_surrogate(self.client_Dx[k], self.client_Dc[k], self.client_Dy[k])
                self.regret[k].append(self._step(k, model))
                c_cand = self.C[t]
                x_cand = self.X[t]
                best_x_mu = []
                for c in c_cand:
                    x_mu, _ = optimize_posterior_mean(model, c, x_cand)
                    best_x_mu.append(torch.tensor(x_mu))
                best_x_mu = torch.stack([x.squeeze() for x in best_x_mu])  # shape: (T,) or (T, 1)
                x_t, c_t, _, _, _, _ = optimize_max_across_contexts(model, best_x_mu, x_cand, c_cand, seed= 20250717) 
                y_t = self.f_ks[k](x_t, c_t) + self.noise_std * torch.randn(1)
                self.client_Dx[k].append(x_t)
                self.client_Dc[k].append(c_t)
                self.client_Dy[k].append(y_t)

        return self._collect_results()
    
    # run random sampling
    def run_rand(self):
        for t in trange(self.t_0 + 1, self.T + 1, desc="Running FMTS_rand"):

            c_cand = self.C[t]
            x_cand = self.X[t]
            for k in range(self.K):
                model = train_gp_surrogate(self.client_Dx[k], self.client_Dc[k], self.client_Dy[k])
                self.regret[k].append(self._step(k, model))
                c = c_cand[torch.randint(0, c_cand.shape[0], (1,)).item()]
                x = x_cand[torch.randint(0, x_cand.shape[0], (1,)).item()]
                y = self.f_ks[k](x, c) + self.noise_std * torch.randn(1)
                self.client_Dx[k].append(x)
                self.client_Dc[k].append(c)
                self.client_Dy[k].append(y)

        return self._collect_results()

    # the proposed method
    def run_fd(self):
        for t in trange(self.t_0 + 1, self.T + 1, desc="Running FMTS_fd"):
            proposed_x = []
            proposed_c = []
            proposed_des = []
            proposed_val = []
            proposed_samp = []
            proposed_idx = []
            for k in range(self.K):    
                model = train_gp_surrogate(self.client_Dx[k], self.client_Dc[k], self.client_Dy[k])
                self.regret[k].append(self._step(k, model))
                c_cand = self.C[t]
                x_cand = self.X[t]
                best_x_mu = []
                for c in c_cand:
                    x_mu, _ = optimize_posterior_mean(model, c, x_cand) 
                    best_x_mu.append(torch.tensor(x_mu))
                
                best_x_mu = torch.stack([x.squeeze() for x in best_x_mu])  # shape: (T,) or (T, 1)
                x_t, c_t, des, des_val, c_idx, samp = optimize_max_across_context_fd(model, best_x_mu, x_cand, c_cand, seed= 20250717) 
                proposed_x.append(x_t)
                proposed_c.append(c_t)
                proposed_des.append(des)
                proposed_val.append(des_val)
                proposed_idx.append(c_idx)
                proposed_samp.append(samp)


                
            
            for k in range(self.K):
                if random.random() > 1/(np.sqrt(t-self.t_0+1)):
                    x_t = proposed_x[k]
                    c_t = proposed_c[k]

                else:
                    x_proposed_cand = proposed_samp[k].argmax(dim=1)
                    stacked = torch.stack(proposed_samp, dim=0)   # shape [K, N, M]
                    samp = stacked.mean(dim=0)  # average over K → shape [N, M]                    samp = proposed_samp.mean(dim=0)   # shape (n_rows, n_cols)

                    row_max = samp.max(dim=1).values
                    rows = torch.arange(samp.size(0))
                    chosen_vals = samp[rows, x_proposed_cand]
                    value_gap = row_max - chosen_vals                 # shape (n_rows,)
                    c_idx = value_gap.argmax()
                    c_t = c_cand[c_idx]
                    x_t = x_cand[samp[c_idx].argmax()]    
                    #x_t = proposed_des[k][c_idx]

                      
                y_t = self.f_ks[k](x_t, c_t) + self.noise_std * torch.randn(1)
                self.client_Dx[k].append(x_t)
                self.client_Dc[k].append(c_t)
                self.client_Dy[k].append(y_t)

                
        return self._collect_results()

    # the proposed method with rff approximation
    def run_fd_rff(self):
        for t in trange(self.t_0 + 1, self.T + 1, desc="Running FMTS_fd"):
            W_shared, b_shared = create_shared_rff_basis(input_dim=(self.c_dim+self.x_dim), num_features=1000, seed=42)
            proposed_x = []
            proposed_c = []
            proposed_des = []
            proposed_val = []
            proposed_samp = []
            proposed_idx = []
            for k in range(self.K):    
                model = train_gp_surrogate(self.client_Dx[k], self.client_Dc[k], self.client_Dy[k])
                self.regret[k].append(self._step(k, model))
                c_cand = self.C[t]
                x_cand = self.X[t]
                best_x_mu = []
                for c in c_cand:
                    x_mu, _ = optimize_posterior_mean(model, c, x_cand) 
                    best_x_mu.append(torch.tensor(x_mu))
                
                best_x_mu = torch.stack([x.squeeze() for x in best_x_mu])  # shape: (T,) or (T, 1)
                x_t, c_t, _, _, _, _ = optimize_max_across_context_fd(model, best_x_mu, x_cand, c_cand, seed= 20250717) 
                proposed_x.append(x_t)
                proposed_c.append(c_t)

                w_hat, ls, os, _ = compute_rff_weights_from_gp(model, W_shared, b_shared)
                _, _, des, des_val, c_idx, samp = optimize_max_across_context_fd_rff(W_shared, b_shared, w_hat, ls, os, x_mu, x_cand, c_cand)

                proposed_des.append(des)
                proposed_val.append(des_val)
                proposed_idx.append(c_idx)
                proposed_samp.append(samp)
                
            for k in range(self.K):
                if random.random() > 1/(np.sqrt(t-self.t_0+1)):
                    x_t = proposed_x[k]
                    c_t = proposed_c[k]

                else:
                    x_proposed_cand = proposed_samp[k].argmax(dim=1)
                    stacked = torch.stack(proposed_samp, dim=0)   # shape [K, N, M]
                    samp = stacked.mean(dim=0)  # average over K → shape [N, M]                    samp = proposed_samp.mean(dim=0)   # shape (n_rows, n_cols)

                    row_max = samp.max(dim=1).values
                    rows = torch.arange(samp.size(0))
                    chosen_vals = samp[rows, x_proposed_cand]
                    value_gap = row_max - chosen_vals                 # shape (n_rows,)
                    c_idx = value_gap.argmax()
                    c_t = c_cand[c_idx]
                    x_t = x_cand[samp[c_idx].argmax()]    
                    #x_t = proposed_des[k][c_idx]

                      
                y_t = self.f_ks[k](x_t, c_t) + self.noise_std * torch.randn(1)
                self.client_Dx[k].append(x_t)
                self.client_Dc[k].append(c_t)
                self.client_Dy[k].append(y_t)
                
        return self._collect_results()
    
    # benchmark method Fedarated thompson sampling   
    def run_fts(self):
        for t in trange(self.t_0 + 1, self.T + 1, desc="Running FMTS_fts"):
            proposed_x = []
            proposed_c = []
            proposed_des = []
            proposed_val = []
            proposed_samp = []
            proposed_idx = []
            for k in range(self.K):    
                model = train_gp_surrogate(self.client_Dx[k], self.client_Dc[k], self.client_Dy[k])
                self.regret[k].append(self._step(k, model))
                c_cand = self.C[t]
                x_cand = self.X[t]
                best_x_mu = []
                for c in c_cand:
                    x_mu, _ = optimize_posterior_mean(model, c, x_cand) 
                    best_x_mu.append(torch.tensor(x_mu))
                
                best_x_mu = torch.stack([x.squeeze() for x in best_x_mu])  # shape: (T,) or (T, 1)
                x_t, c_t, des, des_val, c_idx, samp = optimize_max_across_context_fd(model, best_x_mu, x_cand, c_cand, seed= 20250717) 
                proposed_x.append(x_t)
                proposed_c.append(c_t)
                proposed_des.append(des)
                proposed_val.append(des_val)
                proposed_idx.append(c_idx)
                proposed_samp.append(samp)
                

            for k in range(self.K):
                if random.random() > 1/(np.sqrt(t-self.t_0+1)):
                    x_t = proposed_x[k]
                    c_t = proposed_c[k]

                else:
                    c_idx = torch.randint(0, c_cand.shape[0], (1,)).item()
                    c_t = c_cand[c_idx]
                    x_t = proposed_des[torch.randint(0, self.K, (1,)).item()][c_idx]
                      
                y_t = self.f_ks[k](x_t, c_t) + self.noise_std * torch.randn(1)
                self.client_Dx[k].append(x_t)
                self.client_Dc[k].append(c_t)
                self.client_Dy[k].append(y_t)
                
        return self._collect_results()
    
    def _collect_results(self):
        regret_trace = [sum(self.regret[k][t] / self.K for k in range(self.K)) for t in range(self.T - self.t_0)]
        return self.client_Dx, self.client_Dc, self.client_Dy, self.regret, regret_trace

##############################################################################################################################
