'''
Example 6.5: Windy Gridworld Shown inset below is a standard gridworld, with start and goal states, but with one difference: there is a crosswind running upward through the middle of the grid. 
The actions are the standard four—up, down, right, and left—but in the middle region the resultant next states are shifted upward by a “wind,” the strength of which varies from column to column. 
The strength of the wind is given below each column, in number of cells shifted upward. 
For example, if you are one cell to the right of the goal, then the action left takes you to the cell just above the goal. 
This is an undiscounted episodic task, with constant rewards of -1 until the goal state is reached.
The graph to the right shows the results of applying epsilon-greedy Sarsa to this task, with epsilon = 0.1, alpha = 0.5, and the initial values Q(s, a)=0 for all s, a. 
The increasing slope of the graph shows that the goal was reached more quickly over time. 
By 8000 time steps, the greedy policy was long since optimal (a trajectory from it is shown inset); 
continued epsilon-greedy exploration kept the average episode length at about 17 steps, two more than the minimum of 15. 
Note that Monte Carlo methods cannot easily be used here because termination is not guaranteed for all policies. 
If a policy was ever found that caused the agent to stay in the same state, then the next episode would never end. 
Online learning methods such as Sarsa do not have this problem because they quickly learn during the episode that such policies are poor, and switch to something else.
'''
import torch
import random

epsilon = 0.1
alpha = 0.5

height = 7
width = 10
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

n_actions = 4 # up, down, right, left
actions = [(-1,0),
           (1,0),
           (0,1),
           (0,-1)]
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