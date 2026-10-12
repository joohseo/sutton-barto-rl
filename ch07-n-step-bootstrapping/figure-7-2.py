"""
Reproduction of Figure 7.2: Performance of n-step TD methods as a function of alpha,
for various values of n, on a 19-state random walk task (Example 7.1).

All (alpha, n, run) combinations run as one batched tensor.
"""
import torch
import matplotlib.pyplot as plt


alpha = torch.arange(0, 1, 0.01)
n = 2 ** torch.arange(10)
n_runs = 100
n_state = 19

V = torch.zeros(len(alpha), len(n), n_runs, n_state+2) # (len_a, len_n, n_runs, n_state+2)

initial_state = 10
n_episode = 10


i, j = torch.meshgrid(torch.arange(len(n)), torch.arange(n_runs), indexing='ij')
r = torch.arange(n_runs)

true_values = torch.arange(-0.9, 1, 0.1)
errors = torch.zeros(len(alpha), len(n), n_episode)


for episode in range(n_episode):
    L = 256
    states = None
    while True:
        a = torch.randint(0, 2, (n_runs, L)) * 2 - 1
        if states is not None:
            states = torch.cat([states, states[:, -1:] + torch.cumsum(a, dim=1)], dim=1)
        else:
            states = initial_state + torch.cumsum(a, dim=1)  # (n_runs, L)
        terminal = (states == 0) | (states == n_state + 1)   # (n_runs, ?)
        if terminal.any(dim=1).all():
            break
    states = torch.cat([torch.full((n_runs,1), initial_state), states], dim=1) # concat initial state
    T = terminal.int().argmax(dim=1) + 1   # (n_runs)
    frozen_states = torch.where(torch.arange(terminal.size()[1] + 1)[None,:] >= T[:,None],
                                states.gather(1, T[:, None]), 
                                states)

    rewards = torch.zeros(frozen_states.size())
    rewards[torch.arange(n_runs), T] = torch.full((n_runs,), 1, dtype=torch.float) - 2 * (states.gather(1, T[:, None]) == 0).squeeze(1)
    cumsum_rewards = rewards.cumsum(1)      # frozen_states.size() = (n_runs, ?)

    # loop over tau, not t: a t-loop needs length >= T.max() + max(n), which isn't guaranteed
    for tau in range(int(T.max())):
        tau_n = (tau + n).clamp(max=frozen_states.size()[1] - 1)
        G = cumsum_rewards[:, tau_n].T - cumsum_rewards[None, :, tau]       # (len_n, n_runs) - (1, n_runs) = (len_n, n_runs)
        G = G[None, :, :] + V[:, i, j, frozen_states[:, tau_n].T]           # (len_a, len_n, n_runs)
        TD_n = G - V[:, :, r, frozen_states[:, tau]]                        # (len_a, len_n, n_runs)
        V[:, :, r, frozen_states[:, tau]] += alpha[:, None, None] * TD_n    # (len_a, len_n, n_runs)

    errors[:, : ,episode] = (true_values - V[:, :, :, 1:n_state+1]).pow(2).mean(dim=3).sqrt().mean(dim=2) # (len_a, len_n)

errors = errors.mean(dim=2) # (len_a, len_n)

for n_val in range(len(n)):
    plt.plot(alpha, errors[:,n_val], label=f"n={int(n[n_val])}")
plt.ylim(0.25, 0.55)
plt.xlabel(r"$\alpha$")
plt.ylabel("Average RMS error\n(19 states, first 10 episodes)")
plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("figure-7-2.png", bbox_inches="tight")