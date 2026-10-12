"""
Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size epsilon-greedy algorithm with
alpha = 0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.

Methods compared:
  - epsilon-greedy, sample averages
  - epsilon-greedy, constant step size alpha = 0.1
  - greedy with optimistic initialization, alpha = 0.1
  - UCB, alpha = 0.1
  - gradient bandit with baseline
"""

from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

def bandit(actions, q_star, rng):
    i, j = np.indices(actions.shape)
    return rng.normal(loc=q_star[i, j, actions], scale=1)
     
class Greedy:
    def __init__(self, n_runs, n_params, n_arms, constant, epsilons=None, optimistic_values=None):
        self.n_runs = n_runs
        self.n_params = n_params   
        self.n_arms = n_arms
        self.constant = constant
        self.epsilons = np.asarray(epsilons) if epsilons is not None else None
        self.optimistic_values = np.asarray(optimistic_values) if optimistic_values is not None else None
        self.q_star = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.q_est = np.zeros((self.n_runs, self.n_params, self.n_arms)) if optimistic_values is None else self.optimistic_values[None, :, None] * np.ones((self.n_runs, 1, n_arms))  
        if self.constant:
            self.alpha = 0.1
        else:
            self.action_count = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.reward_data = np.zeros((self.n_runs, self.n_params))
        self.i, self.j = np.indices((n_runs, n_params))
        self.rng = np.random.default_rng(seed=0)

    def update_q_star(self):
        self.q_star += self.rng.normal(0, 0.01, size=(self.n_runs, self.n_params, self.n_arms))

    def update_q_est(self, actions, rewards):
        if self.constant:
            self.q_est[self.i, self.j, actions] += (rewards - self.q_est[self.i, self.j, actions]) * self.alpha
        else:
            self.q_est[self.i, self.j, actions] += (rewards - self.q_est[self.i, self.j, actions]) / self.action_count[self.i, self.j, actions]

    def action_selection(self):
        if self.epsilons is None:
            decision_matrix = np.zeros((self.n_runs, self.n_params)) 
        else:
            decision_matrix = self.rng.random((self.n_runs, self.n_params)) < (self.epsilons[None, :] * np.ones((self.n_runs, 1)))
        random_a = self.rng.integers(0, self.n_arms, size=(self.n_runs, self.n_params))
        greedy = np.argmax(self.q_est, axis=2)
        actions = np.where(decision_matrix, random_a, greedy)
        return actions

    def mean_data(self, measure):
        return np.mean(self.reward_data / measure, axis=0)

    def main_loop(self, warmup, measure):
        for step in range(warmup + measure):
            actions = self.action_selection()
            rewards = bandit(actions, self.q_star, self.rng)

            if not self.constant:
                self.action_count[self.i, self.j, actions] += 1
            self.update_q_est(actions, rewards)

            if step >= warmup:
                self.reward_data += rewards

            self.update_q_star()
            
            if step % 10000 == 0:
                print(f"steps : {step}")
        
class UCB:
    def __init__(self, n_runs, n_params, n_arms, c_values):
        self.n_runs = n_runs
        self.n_params = n_params   
        self.n_arms = n_arms
        self.c_values = np.asarray(c_values)

        self.q_star = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.q_est = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.alpha = 0.1
        self.action_count = np.full((n_runs, n_params, n_arms), 1e-10) # avoid divide by 0

        
        self.reward_data = np.zeros((self.n_runs, self.n_params))
        self.i, self.j = np.indices((n_runs, n_params))
        self.rng = np.random.default_rng(seed=0)

    def update_q_star(self):
        self.q_star += self.rng.normal(0, 0.01, size=(self.n_runs, self.n_params, self.n_arms))

    def update_q_est(self, actions, rewards):
        self.q_est[self.i, self.j, actions] += (rewards - self.q_est[self.i, self.j, actions]) * self.alpha

    def action_selection(self, t):
        ucb_term = self.q_est + self.c_values[None, :, None] * np.sqrt(np.log(t) / self.action_count)
        return np.argmax(ucb_term, axis=2)

    def mean_data(self, measure):
        return np.mean(self.reward_data / measure, axis=0)

    def main_loop(self, warmup, measure):
        for step in range(warmup + measure):
            actions = self.action_selection(step+ 1) # avoid log(0)
            rewards = bandit(actions, self.q_star, self.rng)

            self.update_q_est(actions, rewards)
            self.action_count[self.i, self.j, actions] += 1

            if step >= warmup:
                self.reward_data += rewards
            
            self.update_q_star()
            
            
            if step % 10000 == 0:
                print(f"steps : {step}")

class GradientBandit:
    def __init__(self, n_runs, n_params, n_arms, alpha_values):
        self.n_runs = n_runs
        self.n_params = n_params   
        self.n_arms = n_arms
        self.alpha_values = np.asarray(alpha_values)
    
        self.q_star = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.step_size = 0.1
        self.action_count = np.full((n_runs, n_params, n_arms), 1e-10) # avoid divide by 0
        self.preference = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.pi = np.zeros((self.n_runs, self.n_params, self.n_arms))
        self.r_bar = np.zeros((self.n_runs, self.n_params))


        self.reward_data = np.zeros((self.n_runs, self.n_params))
        self.i, self.j = np.indices((n_runs, n_params))
        self.rng = np.random.default_rng(seed=0)
        self.eye = np.eye(n_arms, dtype=np.int8)
    
    def update_preference(self, actions, rewards):
        update_term = self.alpha_values[None, :, None] * (rewards - self.r_bar)[..., None] * (-self.pi + self.eye[actions])
        self.preference += update_term

    def update_q_star(self):
        self.q_star += self.rng.normal(0, 0.01, size=(self.n_runs, self.n_params, self.n_arms))
        
    def update_r_bar(self, actions, rewards):
        self.r_bar += (rewards - self.r_bar) * self.step_size

    def update_pi(self):
        shifted = self.preference - np.max(self.preference, axis=2, keepdims=True)
        exp = np.exp(shifted)
        self.pi = exp / np.sum(exp, axis=2, keepdims=True)

    def action_selection(self):
        gumbel = -np.log(-np.log(self.rng.random(self.preference.shape)))
        idx = np.argmax(self.preference + gumbel, axis=2)
        return idx

    def mean_data(self, measure):
        return np.mean(self.reward_data / measure, axis=0)

    def main_loop(self, warmup, measure):
        for step in range(warmup + measure):
            actions = self.action_selection()
            rewards = bandit(actions, self.q_star, self.rng)

            self.update_preference(actions, rewards)
            self.update_pi()
            self.action_count[self.i, self.j, actions] += 1
            if step == 0:
                self.r_bar = rewards.copy()
            else:
                self.update_r_bar(actions, rewards)
            
            if step >= warmup:
                self.reward_data += rewards
            
            self.update_q_star()
            
            
            if step % 10000 == 0:
                print(f"steps : {step}")

n_runs = 300
n_arms = 10
warmup = 100000
measure = 100000

# epsilon greedy
epsilon_greedy_params = [1/8192, 1/4096, 1/2048, 1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8]
epsilon_greedy_n_params = len(epsilon_greedy_params)

EG = Greedy(n_runs=n_runs, n_params=epsilon_greedy_n_params, n_arms=n_arms, constant=False, epsilons=epsilon_greedy_params)
EG.main_loop(warmup, measure)
EG_data = EG.mean_data(measure)

plt.plot(epsilon_greedy_params, EG_data, label='EG')

# epsilon greedy with constant-step-size
ConstantEG = Greedy(n_runs=n_runs, n_params=epsilon_greedy_n_params, n_arms=n_arms, constant=True, epsilons=epsilon_greedy_params)
ConstantEG.main_loop(warmup, measure)
CEG_data = ConstantEG.mean_data(measure)

plt.plot(epsilon_greedy_params, CEG_data, label='Constant EG')

# greedy with optimistic initialization
q0_params = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]
optimistic_n_params = len(q0_params)

OptimisticEG = Greedy(n_runs=n_runs, n_params=optimistic_n_params, n_arms=n_arms, constant=True, optimistic_values=q0_params)
OptimisticEG.main_loop(warmup, measure)
OEG_data = OptimisticEG.mean_data(measure)

plt.plot(q0_params, OEG_data, label='Optimistic Greedy')

# UCB
c_values = [4, 8, 16, 32, 64, 128, 256]
ucb_n_params = len(c_values)

UCB = UCB(n_runs=n_runs, n_params=ucb_n_params, n_arms=n_arms, c_values=c_values)
UCB.main_loop(warmup, measure)
UCB_data = UCB.mean_data(measure)

plt.plot(c_values, UCB_data, label='UCB')

# gradient bandit
alpha_values = [2**(-17), 2**(-16), 2**(-15), 1/16384, 1/8192, 1/4096, 1/2048, 1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16]
gradient_bandit_n_params = len(alpha_values)

GB = GradientBandit(n_runs=n_runs, n_params=gradient_bandit_n_params, n_arms=n_arms, alpha_values=alpha_values)
GB.main_loop(warmup, measure)
GB_data = GB.mean_data(measure)

plt.plot(alpha_values, GB_data, label='Gradient Bandit')

# plot
plt.xscale('log')

def format_tick(x):
    f = Fraction(x).limit_denominator(10000000)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

all_x = sorted(set(epsilon_greedy_params) | set(q0_params) | set(c_values) | set(alpha_values))
plt.xticks(all_x, [format_tick(x) for x in all_x], rotation=45)
plt.legend()
plt.savefig('exercise-2-11.png')