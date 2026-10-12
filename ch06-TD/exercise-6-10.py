"""
Exercise 6.10: Stochastic Wind (programming) Re-solve the windy gridworld task with
King's moves, assuming that the effect of the wind, if there is any, is stochastic, sometimes
varying by 1 from the mean values given for each column. That is, a third of the time
you move exactly according to these values, as in the previous exercise, but also a third
of the time you move one cell above that, and another third of the time you move one
cell below that. For example, if you are one cell to the right of the goal and you move
left, then one-third of the time you move one cell above the goal, one-third of the time
you move two cells above the goal, and one-third of the time you move to the goal.
"""
import torch
import random

epsilon = 0.1
alpha = 0.5

height = 7
width = 10
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

n_actions = 8 # up, down, right, left / up right, up left, down right, down left
actions = [(-1,0),
           (1,0),
           (0,1),
           (0,-1),
           (-1,1),
           (-1,-1),
           (1,1),
           (1,-1)]
start_state = (3,0)
goal_state = (3,7)


Q = torch.zeros(height, width, n_actions)

def choose_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    q = Q[state]
    best_action_idx = (q == q.max()).nonzero().flatten().tolist()
    return random.choice(best_action_idx)

def environment(state, action):
    y, x = state
    dy, dx = actions[action]
    stochastic_wind_y = 0 if wind[x] == 0 else random.choice([-1, 0, 1])
    wind_y = -wind[x] + stochastic_wind_y
    new_x = min(max(x + dx, 0), width - 1)
    new_y = min(max(y + dy + wind_y, 0), height - 1)
    new_state = (new_y, new_x)
    done = new_state == goal_state
    return (new_state, -1, done)

def update_Q(s, a):
    new_s, r, done = environment(s, a)
    if done:
        Q[s + (a,)] += alpha * (r - Q[s + (a,)]) # Q(terminal, .) = 0
        return (None, None)
    new_a = choose_action(Q, new_s, epsilon)
    Q[s + (a,)] += alpha * (r + Q[new_s + (new_a,)] - Q[s + (a,)])
    return (new_s, new_a)


n_episode = 15000

for epi in range(n_episode):
    initial_s = start_state
    initial_a = choose_action(Q, initial_s, epsilon)
    s = initial_s
    a = initial_a
    i = 0
    while True:
        i += 1
        new_s, new_a = update_Q(s, a)
        if new_s is None:
            print(f"episode {epi}: {i} steps")
            break
        s = new_s
        a = new_a
    alpha = max(0.02, 0.5 * (1 - epi/n_episode)) # decay alpha to converge under stochastic wind


states = torch.zeros(height, width)
n_eval = 2000
lengths = []
for _ in range(n_eval):
    s = start_state
    i = 0
    states[s] += 1
    while True:
        i += 1
        if i > 2000:      # safety cap
            break
        new_s, r, done = environment(s, choose_action(Q, s, 0))
        states[new_s] += 1
        if done:
            lengths.append(i)
            break
        s = new_s

print(f"mean {sum(lengths)/len(lengths):.2f} steps  (optimal = 10.97)")
print(f"min {min(lengths)} / median {sorted(lengths)[len(lengths)//2]} / max {max(lengths)}")
print(states / n_eval)