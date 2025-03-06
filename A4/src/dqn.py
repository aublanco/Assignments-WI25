# This file contains the implementation of the DQN algorithm.

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dm_control import suite

from dataclasses import dataclass

@dataclass
class HyperParams:
    BATCH_SIZE: int = 512
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: int = 1000
    TAU: float = 0.005
    LR: float = 1e-4


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# set up interactive matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

class DMControlCartPoleWrapper:
    """
    Wraps dm_control's cartpole so that it looks like a standard discrete
    environment. We fix a discrete action set of size=2:
      0 -> -1.0
      1 -> +1.0
    """

    def __init__(self, domain_name="cartpole", task_name="swingup"):
        """
        Initialize the environment with the given domain and task name.

        Args:
            domain_name (str, optional): The name of the domain. Defaults to "cartpole".
            task_name (str, optional): The name of the task. Defaults to "swingup".
        """
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        # Discrete actions for DQN
        self.discrete_actions = np.array([[-1.0], [1.0]], dtype=np.float32)

        # Observe a single reset to determine the observation size
        time_step = self.env.reset()
        obs = self._flatten_observation(time_step.observation)
        self.obs_size = obs.shape[0]

        # We mimic 'action_space.n' from Gym
        self.action_space_n = 2

    def _flatten_observation(self, obs_dict):
        """
        Flatten the observation dictionary into a single numpy array.

        Args:
            obs_dict (dict): The observation dictionary containing state information.

        Returns:
            np.ndarray: A flattened numpy array of state information.
        """
        # Flatten all values in the observation dictionary
        return np.concatenate(
            [val.ravel() for key, val in sorted(obs_dict.items())], axis=0
        )

    def reset(self):
        """
        Reset the environment and return the initial observation.

        Returns:
            np.ndarray: The initial observation after resetting the environment.
        """
        time_step = self.env.reset()
        obs = self._flatten_observation(time_step.observation)
        return obs, {}

    def step(self, action_index):
        """
        Take a step in the environment based on the selected action index.

        Args:
            action_index (int): The index of the action to take.

        Returns:
            tuple: A tuple containing the next observation, reward, done flag, truncated flag, and additional info.
        """
        action = self.discrete_actions[action_index]
        time_step = self.env.step(action)
        reward = time_step.reward if time_step.reward is not None else 0.0
        done = time_step.last()
        truncated = False  # You may define your own cutoff if desired

        if not done:
            obs = self._flatten_observation(time_step.observation)
        else:
            obs = None

        return obs, reward, done, truncated, {}


class ReplayMemory:
    """
    Replay memory to store transitions.
    """

    def __init__(self, capacity: int):
        """Initialize the replay memory.

        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Sample a batch of transitions.

        Args:
            batch_size: The number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """
        Initializes the DQN model.

        Args:
            n_observations (int): The size of the input observation space.
            n_actions (int): The number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Forward pass of the DQN model.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class DQNTrainer:
    def __init__(
        self,
        env: DMControlCartPoleWrapper,
        memory: ReplayMemory,
        device: torch.device,
        params: HyperParams,
        max_steps_per_episode: int = 500,
        num_episodes: int = 50,
    ) -> None:
        """
        Initializes the DQNTrainer with the required components to train a DQN agent.
        """
        self.env = env
        self.policy_net = DQN(env.obs_size, env.action_space_n).to(device)
        self.target_net = DQN(env.obs_size, env.action_space_n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=params.LR, amsgrad=True)
        self.memory = memory
        self.device = device
        self.params = params
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes

        # Track rewards per episode
        self.episode_rewards = []
        # Count the number of steps
        self.steps_done = 0

    def select_action(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using an epsilon-greedy policy based on current Q-network.
        """
        # Compute epsilon threshold
        sample = random.random()
        eps_threshold = (self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) 
                         * math.exp(-1 * self.steps_done / self.params.EPS_DECAY))

        # Update steps
        self.steps_done += 1

        # Exploit or explore
        if sample > eps_threshold: 
            with torch.no_grad():
                # Choose best action from Q-network
                return self.policy_net(state_tensor).max(1).indices.view(1, 1) # provided
        else:
            # Choose random action
            return torch.tensor(
                [[random.randrange(self.env.action_space_n)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self) -> None:
        """
        Performs one gradient descent update on the policy network using a random minibatch sampled from replay memory.
        
        Purpose:
          1. Sampling the Batch:
             - Sample a minibatch of transitions (s, a, r, s') from the replay memory.
          2. Batch Processing:
             - Unpack transitions into batches.
             - Prepare tensors for current states, actions, rewards, and non-terminal next states.
          3. Q-value Calculation:
             - Current Q-values: For each state s in the batch, compute Q(s, a; θ) using the policy network and select Q(s, a) via gather.
             - Next Q-values: For non-terminal states, compute max_a' Q(s', a'; θ⁻) using the target network; terminal states yield 0.
          4. Target Computation:
             - Compute the target Q-values as: y = r + γ * max_a' Q(s', a'; θ⁻).
          5. Loss Calculation:
             - Compute the loss using Smooth L1 (Huber) loss between current Q-values and the target.
          6. Optimization:
             - Zero gradients, backpropagate the loss, clip gradients if necessary, and take an optimizer step.
        """
        # STEP 1: Check if there's enough data in replay memory; if not, simply return.
        if len(self.memory) < self.params.BATCH_SIZE:
            return
        # STEP 2: Sample a minibatch of transitions from replay memory.
        transitions = self.memory.sample(self.params.BATCH_SIZE)
        # STEP 3: Unpack transitions into batches for state, action, reward, and next_state.
        batch = Transition(*zip(*transitions))
        # STEP 4: Prepare masks and tensors:
        
        non_final_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([x for x in batch.next_state if x is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # STEP 5: Calculate current Q-values:
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # STEP 6: Compute next state Q-values for non-terminal states:
        next_state_values = torch.zeros(self.params.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # STEP 7: Compute the target Q-values using the Bellman equation:
        expected_state_action_values = reward_batch + (self.params.GAMMA * next_state_values)
        # STEP 8: Compute loss using Smooth L1 (Huber) loss:
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # STEP 9: Optimize the model:
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self) -> None:
        """
        #### 4. Target Network Update: `soft_update`    
        **Purpose:**
        Performs a soft update of the target network parameters to slowly track the policy network.
        
        **Mathematical Formulation:**
        
        - For each parameter θ_target in the target network:
        
          θ_target ← τ θ_policy + (1 - τ) θ_target
        
          where τ is a small constant (e.g., 0.005) that determines the update rate. This gradual adjustment helps stabilize training.
        
        **Implementation Details:**
        
        - Both the target and policy network state dictionaries are iterated over, and the update is applied element-wise.
        - Ensure that the parameters are moved to the appropriate device (CPU/GPU) prior to the update.
        """
        with torch.no_grad():
            target_net_dict = self.target_net.state_dict()
            policy_net_dict = self.policy_net.state_dict()

            for key in policy_net_dict:
                target_net_dict[key] = (self.params.TAU * policy_net_dict[key] + 
                                        (1 - self.params.TAU) * target_net_dict[key])
                
            self.target_net.load_state_dict(target_net_dict)
        

    def plot_rewards(self, show_result: bool = False) -> None:
        """
        Plots accumulated rewards for each episode.
        """
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)

        # Decide whether to clear figure or show final result
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training (Reward)")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(rewards_t.numpy(), label="Episode Reward")

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def train(self) -> None:
        """
        Runs the main training loop across the specified number of episodes.

        Guidance for implementation:
        1. For each episode, reset the environment and initialize the state and episode_reward.
        2. For each time step within the episode:
           - Use self.select_action() to choose an action based on the current state.
           - Apply the action in the environment using self.env.step() to obtain the next observation, reward, and termination status.
           - Convert the observation to a tensor (if the episode hasn't ended) to form the next_state.
           - Store the transition (state, action, next_state, reward) in the replay memory using self.memory.push().
           - Perform an optimization step with self.optimize_model() to update the policy network.
           - Update the target network with self.soft_update().
           - Break the loop if the episode has ended.
        3. After each episode, record the accumulated episode reward and update the rewards plot by calling self.plot_rewards().
        4. After all episodes, print "Training complete", display the final rewards plot, and close the interactive plot.

        Note: Retain the plot_rewards() related logic for tracking and visualizing training progress.
        """
        for _ in range(self.num_episodes):
            # TODO: Reset the environment and initialize state and episode_reward.
            obs, info = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_reward = 0.0

            for _ in range(self.max_steps_per_episode):
                action = self.select_action(state)
                obs, reward, term, trunc, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = term or trunc

                if term: 
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                
                state = next_state

                self.optimize_model()

                self.soft_update()

                episode_reward += reward.item()

                if done:
                    break

            # Tracking episode reward and plotting rewards.
            self.episode_rewards.append(episode_reward)
            self.plot_rewards()

        print("Training complete")
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()
        plt.savefig("rewards_plot_dqn.png")
