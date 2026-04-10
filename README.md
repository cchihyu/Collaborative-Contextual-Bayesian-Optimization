# Collaborative-Contextual-Bayesian-Optimization

**Abstract**
Discovering optimal designs through sequential data collection is essential in many real-world applications. While Bayesian Optimization (BO) has achieved remarkable success in optimal design, growing attention has recently turned to context-specific optimal design, a problem that has been formalized as Contextual Bayesian Optimization (CBO). Unlike standard BO, CBO is inherently more challenging, as it aims to approximate a mapping from the entire context space to its corresponding optimal design. This complexity requires efficient exploration across contexts and effective balancing of exploration and exploitation within each context. Fortunately, in many modern applications, optimal design tasks arise across multiple related clients or experimental units, where collaboration can significantly improve learning efficiency. Leveraging these collaborative benefits, we propose \texttt{CCBO}, Collaborative Contextual Bayesian Optimization, a unified framework that enables multiple clients to jointly perform CBO under settings where the context is controllable. We establish theoretical guarantees and demonstrate, through extensive simulations and real-world applications, that the proposed framework achieves substantial improvements over existing approaches.

## Test Functions

Experiments are run on standard BO benchmark functions, split into design variable $x$ and context variable $c$:

- **Branin** (1D × 1D)
- **Levy** (configurable dimension)
- **Ackley** (configurable dimension)
- **Hartmann** (2D × 2D)

Client-specific functions $f_k$ are constructed by applying small random perturbations (shifts and scales) to a shared base function via `create_perturbed_f_k`.

## Regret

Regret is computed as the normalized gap between the per-context oracle maximum and the current posterior-mean-optimal design, averaged across contexts and normalized by the range of the function:

$$R_t = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{f^*(c) - f(\hat{x}_t(c), c)}{\text{normalizing constant}}$$

where $\hat{x}_t(c) = \arg\max_x \mu_t(x, c)$ is the greedy design under the current GP posterior mean.

## Dependencies

- [PyTorch](https://pytorch.org/)
- [BoTorch](https://botorch.org/)
- [GPyTorch](https://gpytorch.ai/)
- [tqdm](https://github.com/tqdm/tqdm)
- [pandas](https://pandas.pydata.org/)

Install with:

```bash
pip install torch botorch gpytorch tqdm pandas
```

## Usage

```python
from exp_class import FederatedThompsonSampler
from helper_CMTS import branin, create_perturbed_f_k

x_dim, c_dim = 1, 1
K = 3   # number of clients
T = 50  # optimization rounds
t_0 = 5 # initial random rounds

f_ks = [create_perturbed_f_k(branin, x_dim, c_dim) for _ in range(K)]

sampler = FederatedThompsonSampler(
    f=branin, x_dim=x_dim, c_dim=c_dim,
    K=K, T=T, t_0=t_0, f_ks=f_ks, seed=42
)

# Run the proposed federated method
Dx, Dc, Dy, regret, regret_trace = sampler.run_fd()

# Reset and run a baseline
sampler.reset_data()
Dx, Dc, Dy, regret, regret_trace = sampler.run_indep()
```

Results can be exported to Excel/CSV using `export_experiment_results` from `helper_CMTS.py`.
