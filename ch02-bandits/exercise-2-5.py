"""
Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for nonstationary problems.
Use a modified version of the 10-armed testbed in which all the q*(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the q*(a) on each step). 
Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, 
and another action-value method using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and longer runs, say of 10,000 steps.
"""

import numpy as np
import matplotlib.pyplot as plt

def bandit(action, q_star):
    return np.random.normal(loc=q_star[action], scale=1)

def action_selection(epsilon, q_est):
    if np.random.rand() < epsilon:
        action = np.random.choice(10)
    else:
        max_idx = np.where(q_est == np.max(q_est))[0]
        action = np.random.choice(max_idx)
    return action

def update_q_star(q_star):
    q_star += np.random.normal(0, 0.01, size=10)
    return q_star

def update_q_est_incremental(q_est, action, reward, action_count):
    q_est[action] = q_est[action] + (reward-q_est[action]) / action_count[action]

def update_q_est_constant(q_est, action, reward, alpha):
    q_est[action] = q_est[action] + (reward-q_est[action]) * alpha

def sim_single_run(steps):
    reward_inc_list = np.zeros(steps)
    reward_con_list = np.zeros(steps)
    optimal_inc_list = np.zeros(steps)
    optimal_con_list = np.zeros(steps)

    q_star = np.zeros(10)
    q_est_inc = np.zeros(10)
    q_est_con = np.zeros(10)
    action_count = np.zeros(10)

    epsilon = 0.1
     
    for step in range(steps):
        action_inc = action_selection(epsilon, q_est_inc)
        reward_inc = bandit(action_inc, q_star)
        action_count[action_inc] += 1
        update_q_est_incremental(q_est_inc, action_inc, reward_inc, action_count)
        reward_inc_list[step] = reward_inc
        optimal_action = np.argmax(q_star)
        optimal_inc_list[step] = 1 if action_inc == optimal_action else 0


        action_con = action_selection(epsilon, q_est_con)
        reward_con = bandit(action_con, q_star)
        update_q_est_constant(q_est_con, action_con, reward_con, alpha=0.1)
        reward_con_list[step] = reward_con
        optimal_con_list[step] = 1 if q_star[action_con] >= np.max(q_star) else 0

        update_q_star(q_star)
    
    return np.stack([reward_inc_list, reward_con_list, optimal_inc_list, optimal_con_list], axis=0)

def average_performance(runs, steps):
    average_array = np.zeros((4, steps))
    for run in range(runs):
        data = sim_single_run(steps)
        average_array += data
        if run % 100 == 0:
            print(f"runs : {run}")
    return average_array / runs
    
data = average_performance(runs=2000, steps=10000)

plt.subplot(2, 1, 1)
plt.plot(data[0,:], label='reward_inc')
plt.plot(data[1,:], label='reward_con')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data[2,:], label='action_inc')
plt.plot(data[3,:], label='action_con')
plt.legend()

plt.show()
