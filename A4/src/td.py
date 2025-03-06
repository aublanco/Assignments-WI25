import numpy as np
from tqdm import tqdm
from .discretize import quantize_state, quantize_action
from dm_control.rl.control import Environment

# Initialize the Q-table with dimensions corresponding to each discretized state variable.
def initialize_q_table(state_bins: dict, action_bins: list) -> np.ndarray:
    """
    Initialize the Q-table with dimensions corresponding to each discretized state variable.

    Args:
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.

    Returns:
        np.ndarray: A Q-table initialized to zeros with dimensions matching the state and action space.
    """
    state_shape = tuple(
        len(np.asarray(b).flatten()) + 1
        for key, bins in sorted(state_bins.items())
        for b in bins
    )
    
    q_shape = state_shape + (len(action_bins),)
    q_table = np.zeros(q_shape, dtype = float)
    return q_table

# TD Learning algorithm
def td_learning(env: Environment, num_episodes: int, alpha: float, gamma: float, epsilon: float, state_bins: dict, action_bins: list, q_table:np.ndarray=None) -> tuple:
    """
    TD Learning algorithm for the given environment.

    Args:
        env (Environment): The environment to train on.
        num_episodes (int): The number of episodes to train.
        alpha (float): The learning rate. 
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.
        q_table (np.ndarray): The Q-table to start with. If None, initialize a new Q-table.

    Returns:
        tuple: The trained Q-table and the list of total rewards per episode.
    """
    if q_table is None:
        q_table = initialize_q_table(state_bins, action_bins)
        
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        time_step = env.reset()
        state_index = quantize_state(time_step.observation, state_bins)
        done = False
        ep_reward = 0

        if np.random.rand() < epsilon:
            action = np.random.choice(action_bins)
        else: 
            best_action = int(np.argmax(q_table[state_index]))
            action = action_bins[best_action]
        action_index = quantize_action(action, action_bins)

        while not done:
            time_step = env.step(action_bins[action_index])
            next_state_index = quantize_state(time_step.observation, state_bins)
            reward = time_step.reward

            max_next_q = np.max(q_table[next_state_index])

            q_table[state_index + (action_index,)] += alpha*(
                reward + gamma * max_next_q - q_table[state_index + (action_index,)] 
                - q_table[state_index + (action_index,)]
            )
            
            state_index = next_state_index

            if np.random.rand() < epsilon:
                action = np.random.choice(action_bins)
            else:
                best_action = int(np.argmax(q_table[state_index]))
                action = action_bins[best_action]
            action_index = quantize_action(action, action_bins)
            
            ep_reward += reward

            done = time_step.last()

        rewards.append(ep_reward)

    return q_table, rewards


def greedy_policy(q_table: np.ndarray) -> callable:
    """
    Define a greedy policy based on the Q-table.    

    Args:
        q_table (np.ndarray): The Q-table from which to derive the policy.

    Returns:
        callable: A function that takes a state and returns the best action. 
    """
    def policy(state):
        return np.argmax(q_table[state])
    return policy