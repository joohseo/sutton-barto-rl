"""
Exercise 5.12: Racetrack (programming)
Consider driving a race car around a turn like those shown in Figure 5.5.
You want to go as fast as possible, but not so fast as to run off the track.
In our simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram.
The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step.
The actions are increments to the velocity components. Each may be changed by +1, -1, or 0 in each step, for a total of nine (3x3) actions.
Both velocity components are restricted to be nonnegative and less than 5, and they cannot both be zero except at the starting line.
Each episode begins in one of the randomly selected start states with both velocity components zero and ends when the car crosses the finish line.
The rewards are -1 for each step until the car crosses the finish line.
If the car hits the track boundary, it is moved back to a random position on the starting line, both velocity components are reduced to zero, and the episode continues.
Before updating the car's location at each time step, check to see if the projected path of the car intersects the track boundary.
If it intersects the finish line, the episode ends; if it intersects anywhere else, the car is considered to have hit the track boundary and is sent back to the starting line.
To make the task more challenging, with probability 0.1 at each time step the velocity increments are both zero, independently of the intended increments.
Apply a Monte Carlo control method to this task to compute the optimal policy from each starting state.
Exhibit several trajectories following the optimal policy (but turn the noise off for these trajectories).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

class OffPolicyMC:
    def __init__(self, positions, max_velocity=4):
        self.positions = positions # 0: outside of the track, 1: start line, 2: track, 3: finish line
        self.max_velocity = max_velocity
        self.rng = np.random.default_rng(seed=0)
        i, j = np.indices((3,3))
        self.actions = np.concatenate((i[..., None]-1, j[..., None]-1), axis=-1).reshape(-1, 2)
        row, col = positions.shape
        self.q = -100 * self.rng.random(size=(row, col, max_velocity + 1, max_velocity + 1, len(self.actions)))
        self.valid, self.invalid = self.masks()
        self.mask_q()
        self.count = np.zeros((row, col, max_velocity + 1, max_velocity + 1, len(self.actions)))
        self.greedy_policy = np.argmax(self.q, axis=-1)
        self.gamma = 1

    def position_to_idx(self, position):
        x, y = position
        if x >= self.q.shape[1] or y >= self.q.shape[0]:
            raise
        return (self.q.shape[0] - y - 1, x)

    def state_to_idx(self, state):
        x, y, v_x, v_y = state
        state_idx = self.position_to_idx((x, y)) + (v_x, v_y)
        return state_idx

    def idx_to_position(self, idx):
        i, j = idx
        if i >= self.q.shape[0] or j >= self.q.shape[1]:
            raise
        return (j, self.q.shape[0] - i - 1)
        
    def masks(self):
        x, y, v_x, v_y, a = self.q.shape
        vx = np.arange(v_x).reshape(-1, 1, 1)    # (v_x, 1,  1)
        vy = np.arange(v_y).reshape( 1,-1, 1)    # ( 1, v_y, 1)
        ax = self.actions[:, 0].reshape(1, 1,-1) # ( 1,  1,  a)
        ay = self.actions[:, 1].reshape(1, 1,-1) # ( 1,  1,  a)

        new_vx = vx + ax
        new_vy = vy + ay
        
        # invalid.shape : (v_x, v_y, a)
        invalid = (new_vx < 0) | (new_vx > self.max_velocity) | (new_vy < 0) | (new_vy > self.max_velocity) | ((new_vx==0) & (new_vy==0))
        valid = ~invalid
        return (valid, invalid)

    def mask_q(self):
        self.q[:, :, self.invalid] = -np.inf

    def behavior_policy(self, state, epsilon):
        _, _, v_x, v_y = state
        state_idx = self.state_to_idx(state)
        greedy_action_idx = int(self.greedy_policy[state_idx])
        choices = np.flatnonzero(self.valid[v_x, v_y])
        probability = epsilon / len(choices)
        if self.rng.random() < epsilon:
            action_idx = int(self.rng.choice(choices))
            if action_idx == greedy_action_idx:
                probability += (1 - epsilon)
        else:
            action_idx = greedy_action_idx
            probability += (1 - epsilon)
        return (action_idx, probability)


    def choose_start_position(self):
        idxs = np.argwhere(self.positions == 1)
        position_idx = idxs[self.rng.integers(len(idxs))]
        return self.idx_to_position(tuple(position_idx.tolist()))

    def environment(self, state, action_idx, noise):
        x, y, v_x, v_y = state
        if self.rng.random() < noise:
            a_x, a_y = (0, 0)
        else:
            a_x, a_y = self.actions[action_idx]
        v_x = int(v_x + a_x)
        v_y = int(v_y + a_y)

        # collision detection
        path_cells = []
        if v_x == 0:
            for j in range(v_y):
                path_cells.append((x, y + j + 1))
        else:
            slope = v_y / v_x
            x_locs = np.arange(0.5, v_x, 1) # length : v_x
            intersections = slope * x_locs + 0.5
            lower = np.append([0], np.ceil(intersections) - 1, axis=0)
            upper = np.append(np.floor(intersections), [v_y], axis=0)
            for i in range(len(lower)):
                for j in range(int(lower[i]), int(upper[i]) + 1):
                    path_cells.append((x + i, y + j))

        # update reward, state
        reward = -1
        next_state = (x + v_x, y + v_y, v_x, v_y)

        # collision detection override
        for cell in path_cells:
            try:
                idx = self.position_to_idx(cell)
                value = self.positions[idx]
                if value == 0:
                    next_state = self.choose_start_position() + (0, 0)
                    break
                if value == 3:
                    next_state = None
                    break
            except:
                next_state = self.choose_start_position() + (0, 0)
                break           
        return (reward, next_state)

    def gen_episode(self, epsilon=0.1, is_greedy=False, max_steps=10000):
        episode = []
        probabilities = []
        initial_state = self.choose_start_position() + (0, 0) # position, velocity
        state = initial_state
        steps = 0
        while True:
            if is_greedy:
                state_idx = self.state_to_idx(state)
                action_idx = int(self.greedy_policy[state_idx])
                probability = None
                noise = 0
            else:
                action_idx, probability = self.behavior_policy(state, epsilon)
                noise = 0.1
            reward, next_state = self.environment(state, action_idx, noise)
            episode.append((state, action_idx, reward))
            probabilities.append(probability)
            if next_state is None:
                break
            state = next_state
            steps += 1
            # cap greedy episodes in case the policy loops
            if is_greedy and steps >= max_steps:
                print(f"greedy episode truncated: did not finish within {max_steps} steps")
                break
        return (episode, probabilities)

    def main_loop(self, num_episodes):
        for num in range(num_episodes):
            if (num + 1) % 10 == 0:
                print(f"simulated {num + 1} episodes")
            epsilon = 0.6 - (num / num_episodes) * 0.5
            episode, probabilities = self.gen_episode(epsilon)
            g = 0
            weight = 1
            for t, step in reversed(list(enumerate(episode))):
                state, action_idx, reward = step
                state_idx = self.state_to_idx(state)
                state_action_idx = state_idx + (action_idx,)

                g = self.gamma * g + reward
                self.count[state_action_idx] += weight
                self.q[state_action_idx] += (weight / self.count[state_action_idx]) * (g - self.q[state_action_idx])
                self.greedy_policy[state_idx] = np.argmax(self.q[state_idx])
                if action_idx != self.greedy_policy[state_idx]:
                    break
                weight = weight / probabilities[t]
    
    def plot_episode(self, episode, ax=None):
        show = ax is None
        if show:
            _, ax = plt.subplots(figsize=(4, 8))
        cmap = ListedColormap(['black', 'limegreen', 'lightgray', 'red'])  # 0,1,2,3
        norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(self.positions, cmap=cmap, norm=norm)

        seg_r, seg_c = [], []
        for (x, y, vx, vy), a, r in episode:
            i, j = self.position_to_idx((x, y))
            if (vx, vy) == (0, 0) and seg_r:        # reset: start a new segment
                ax.plot(seg_c, seg_r, '-o', color='tab:blue', ms=3)
                seg_r, seg_c = [], []
            seg_r.append(i); seg_c.append(j)
        ax.plot(seg_c, seg_r, '-o', color='tab:blue', ms=3)
        ax.set_title(f"steps = {len(episode)}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if show:
            plt.show()

    def plot_trajectories(self, ncols=5, nrows=2):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 5 * nrows))
        for ax in axes.flat:
            episode, _ = self.gen_episode(is_greedy=True)
            self.plot_episode(episode, ax=ax)
        fig.tight_layout()
        plt.show()


def make_racetrack():
    """Right track of Figure 5.5. 0: outside, 1: start, 2: track, 3: finish."""
    rows, cols = 30, 32
    track = np.zeros((rows, cols), dtype=int)

    def left_bound(r):   # left wall recedes going up
        if r >= 26: return 4
        if r >= 22: return 3
        if r >= 18: return 2
        if r >= 14: return 1
        return 0

    def right_bound(r):  # top 9 rows open to the finish line
        return cols - 1 if r <= 8 else 16

    for r in range(rows):
        track[r, left_bound(r):right_bound(r) + 1] = 2

    track[rows - 1, left_bound(rows - 1):right_bound(rows - 1) + 1] = 1  # start line
    track[0:9, cols - 1] = 3                                            # finish line
    return track


track = make_racetrack()
print(track)

MCmodel = OffPolicyMC(track)
MCmodel.main_loop(200000)
MCmodel.plot_trajectories(ncols=5, nrows=2)
