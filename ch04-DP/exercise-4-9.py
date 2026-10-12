"""
Exercise 4.9 (programming) Implement value iteration for the gambler’s problem and
solve it for ph = 0.25 and ph = 0.55. In programming, you may find it convenient to
introduce two dummy states corresponding to termination with capital of 0 and 100,
giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3.
Are your results stable as theta -> 0?
"""

import numpy as np
import matplotlib.pyplot as plt

def value_iteration(goal=100, ph=0.4, theta=1e-10, round_dec=5):
    states = np.arange(1, goal, dtype=int) # [1, 2, ..., goal-1]
    v = np.zeros((goal + 1)) # v[0]=0, v[1], ..., v[goal-1], v[goal]=1
    v[goal] = 1
    policy = np.zeros((goal)) # policy[0(dummy)], policy[1], ..., policy[goal-1]
    value_list = []
    q_matrix = np.zeros((goal // 2, goal - 1)) # v(state|action) = q_matrix[action-1, state-1]

    i = 1
    while True:
        delta = 0
        for state in states:
            v_temp = v[state]
            actions = np.arange(1, np.minimum(state, goal - state) + 1) # [1, ..., min()]
            v_list = ph * v[state + actions] + (1-ph) * v[state - actions]
            v[state] = np.max(v_list)
            delta = np.maximum(delta, np.abs(v_temp - v[state]))
        
        value_list.append(v.copy())
        print(f"Sweep {i} finished. Delta : {delta}")
        i += 1
        if delta < theta:
            print("Done.")
            break

    for state in states:
        actions = np.arange(1, np.minimum(state, goal - state) + 1) # [1, ..., min()]
        v_list = ph * v[state + actions] + (1-ph) * v[state - actions]
        v_list_round = np.round(v_list, round_dec)

        # Optimal policy
        policy[state] = np.argmax(v_list_round) + 1 # tie-break with rounding

        # q_matrix normalization
        if np.max(v_list - np.min(v_list)) == 0:
            v_list_normalized = np.ones(v_list.shape)
        else:
            v_list_normalized = (v_list - np.min(v_list)) / np.max(v_list - np.min(v_list))

        for idx, action in enumerate(actions):
            q_matrix[action-1, state-1] = v_list_normalized[idx]
    return (states, value_list, policy, q_matrix)


states, value_list, policy, q_matrix = value_iteration(ph=0.4)


# plot
fig, axes = plt.subplot_mosaic(
    [['top_left',    'right'],
     ['bottom_left', 'right']],
    figsize=(8, 6),
    width_ratios=[1, 1.3],
    constrained_layout=True,
)

for value in value_list:
    axes['top_left'].plot(states, value[states])
axes['top_left'].set_xlabel('Capital')
axes['top_left'].set_ylabel('Value estimates')

axes['bottom_left'].bar(states, policy[states])
axes['bottom_left'].set_xlabel('Capital')
axes['bottom_left'].set_ylabel('Final policy (stake)')

axes['right'].matshow(q_matrix, origin='lower',
                      extent=(0.5, q_matrix.shape[1] + 0.5, 0.5, q_matrix.shape[0] + 0.5))
axes['right'].set_xlabel('Capital')
axes['right'].set_ylabel('Action')

plt.show()