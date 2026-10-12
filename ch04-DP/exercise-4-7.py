'''
Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
rental problem with the following changes. One of Jack’s employees at the first location
rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs $2, as do all cars
moved in the other direction. In addition, Jack has limited parking space at each location.
If more than 10 cars are kept overnight at a location (after any moving of cars), then an
additional cost of $4 must be incurred to use a second parking lot (independent of how
many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
occur in real problems and cannot easily be handled by optimization methods other than
dynamic programming.
To check your program, first replicate the results given for the original problem.
'''

import numpy as np
import matplotlib.pyplot as plt
import math

class Poisson:
    def __init__(self, lam, num_cars):
        self.lam = lam
        self.num_cars = num_cars
        self.n_cars_dist = self.dist(self.num_cars)
        
    def prob(self, n): # P(0..n)
        arr = np.arange(n+1)
        factorial = np.zeros(n+1)
        for i in range(n+1):
            factorial[i] = math.factorial(i)
        return (self.lam ** arr) * np.exp(-self.lam) / factorial
    
    def dist(self, n): # P(0..n-1), P(>=n)
        dist = np.zeros((n + 1))
        dist[0:n] = self.prob(n-1)
        dist[n] = 1 - np.sum(dist)
        return dist

num_cars = 20
rent_credit = 10
move_cost = 2
gamma = 0.9

p_req_1 = Poisson(3, num_cars)
p_req_2 = Poisson(4, num_cars)
p_rtrn_1 = Poisson(3, num_cars)
p_rtrn_2 = Poisson(2, num_cars)

prob_matrix_1 = p_req_1.n_cars_dist[:, None] * p_rtrn_1.n_cars_dist[None, :]
prob_matrix_2 = p_req_2.n_cars_dist[:, None] * p_rtrn_2.n_cars_dist[None, :]
prob_matrix = prob_matrix_1[...,None, None] * prob_matrix_2[None, None, ...]

i, j = np.indices((num_cars + 1, num_cars + 1))
states = np.stack((i, j), axis=-1)

v = np.zeros((num_cars + 1, num_cars + 1))
policy = np.zeros((num_cars + 1, num_cars + 1), dtype=int)
policy_list = []
policy_list.append(policy.copy())

max_action = 5

theta = 0.5
epsilon_tol = 0.3

def trunc_prob_matrix(state):
    n1, n2 = state
    tr_prob_matrix = np.concatenate((prob_matrix[0:n1, :, :, :],np.sum(prob_matrix[n1:, :, :, :],axis=0, keepdims = True)),axis=0)
    tr_prob_matrix = np.concatenate((tr_prob_matrix[:, :, 0:n2, :],np.sum(tr_prob_matrix[:, :,n2:,  :],axis=2, keepdims = True)),axis=2)
    return tr_prob_matrix

def calc_state_reward(state, action, modified):
    n1, n2 = state
    if modified:
        max_free_car = 10
        parking_cost = 4
        r_action = -move_cost * (np.abs(action - 1/2) - 1/2) - np.sum((np.array(state) > max_free_car)) * parking_cost
    else:
        r_action = -move_cost * np.abs(action)
    n_req_1 = np.arange(n1 + 1)[:, None, None, None] 
    n_req_2 = np.arange(n2 + 1)[None, None, :, None]

    n_rtrn_1 = np.arange(num_cars + 1)[None, :, None, None]
    n_rtrn_2 = np.arange(num_cars + 1)[None, None, None, :]

    s_1 = np.minimum(n1 - n_req_1 + n_rtrn_1, num_cars)
    s_2 = np.minimum(n2 - n_req_2 + n_rtrn_2, num_cars)
    reward = rent_credit * (n_req_1 + n_req_2) + r_action
    return (s_1, s_2, reward)

def calc_v(state, action, modified):
    n1, n2 = state
    state_after_action = [np.minimum(num_cars, n1-action), np.minimum(num_cars, n2+action)]
    tr_prob_matrix = trunc_prob_matrix(state_after_action)
    s_1, s_2, reward = calc_state_reward(state_after_action, action, modified)
    new_v = np.sum(tr_prob_matrix * (reward + gamma * v[s_1, s_2]))
    return new_v

def policy_evaluation(theta, modified):
    l=1
    while True:
        delta = 0
        for state in states.reshape(-1, 2):
            v_temp = v[tuple(state)]
            action = policy[tuple(state)]
            v[tuple(state)] = calc_v(state, action, modified)
            delta = np.maximum(delta, np.abs(v[tuple(state)] - v_temp))
        print(f"policy evaluation loop : {l}, delta = {delta}")
        l += 1
        if delta < theta:
            break

def main_loop(modified=False):
    while True:
        print("Evaluating policy...")
        policy_evaluation(theta, modified)
        print("Done, now improving policy...")
        policy_stable = True
        for state in states.reshape(-1, 2):
            actions = np.arange(-np.minimum(max_action, state[1]), np.minimum(max_action,state[0]) + 1)
            v_list = np.zeros(actions.shape)
            for idx, action in enumerate(actions):
                v_list[idx] = calc_v(state, action, modified)
            policy[tuple(state)] = actions[np.argmax(v_list)]
            greedy_v = np.max(v_list)
            if np.abs(greedy_v - v[tuple(state)]) > epsilon_tol:
                policy_stable = False
        policy_list.append(policy.copy())
        if policy_stable == True:
            break
        print("Policy not stable.")
    print("Policy stable, Done.")


main_loop(modified=True)


# plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# shared color range
vmin = min(a.min() for a in policy_list)
vmax = max(a.max() for a in policy_list)

for i, ax in enumerate(axes.flat):
    if i < len(policy_list):
        im = ax.matshow(policy_list[i], origin='lower', vmin=vmin, vmax=vmax)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        ax.set_xlabel('#Cars at second location')
        ax.set_ylabel('#Cars at first location')
        ax.set_title(f'Policy {i}')
    else:
        ax.axis('off')

# shared colorbar
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)

plt.show()

