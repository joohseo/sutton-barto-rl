"""
Exercise 6.9: Windy Gridworld with King's Moves (programming) Re-solve the windy
gridworld assuming eight possible actions, including the diagonal moves, rather than four.
How much better can you do with the extra actions? Can you do even better by including
a ninth action that causes no movement at all other than that caused by the wind?
"""
import torch
import random

epsilon = 0.1
alpha = 0.5

height = 7
width = 10
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# king's moves, optionally + no-move
include_no_move = True
actions = [(-1,0),   # up
           (1,0),    # down
           (0,1),    # right
           (0,-1),   # left
           (-1,1),   # up-right
           (-1,-1),  # up-left
           (1,1),    # down-right
           (1,-1)]   # down-left
if include_no_move:
    actions.append((0,0))
n_actions = len(actions)
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
    wind_y = -wind[x]
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


n_episode = 1500

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


states = torch.zeros(height, width)
s = start_state
a = choose_action(Q, s, 0)
i = 0
states[s] += 1
while True:
    i += 1
    new_s, r, done = environment(s, a)
    states[new_s] += 1
    if done:
        break
    new_a = choose_action(Q, new_s, 0)
    s = new_s
    a = new_a

print(f"Greedy policy path:")
print(states)
print(f"steps: {i}")