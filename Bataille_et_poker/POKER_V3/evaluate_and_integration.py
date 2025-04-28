import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Import previously defined modules.
# Ensure that env.py, state_representation.py, and model_and_ppo.py are in the same folder.
from env import SimpleHUNLEnv
from state_representation import combine_card_channels, encode_action_history
from model_and_ppo import AlphaHoldemNet, compute_trinal_clip_policy_loss, compute_trinal_clip_value_loss, compute_entropy_loss

# ------------------------------------------------------------------------------
# Step 6: Logging, Plotting, and Evaluation Functions
# ------------------------------------------------------------------------------

def evaluate_agent(agent, env, num_episodes=500, device=torch.device("cpu")):
    """
    Run the agent in evaluation mode (deterministic) for a number of episodes.
    Returns:
      avg_reward: Average reward over the evaluation episodes.
      rewards: List of rewards per episode.
    """
    agent.model.eval()  # Set model to evaluation mode.
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        # Get the card tensor from the observation.
        agent_hole = obs["hole_cards"].tolist()         # List of 2 integers.
        community_cards = obs["community_cards"].tolist() # List of 5 integers.
        card_tensor = combine_card_channels(agent_hole, community_cards)
        card_tensor = card_tensor.unsqueeze(0).to(device)  # Shape: (1, 7, 4, 13)
        
        # In evaluation, we assume no previous actions; use a zero history.
        history_tensor = torch.zeros((1, 6, 3), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            logits, value = agent.model(card_tensor, history_tensor)
            # For evaluation, use greedy action (deterministic).
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()

        # Execute action in the environment.
        _, reward, done, info = env.step(action)
        episode_rewards.append(reward)
    
    avg_reward = np.mean(episode_rewards)
    agent.model.train()  # Reset back to training mode.
    return avg_reward, episode_rewards

# ------------------------------------------------------------------------------
# Step 7: Integration and Final Testing (Training + Evaluation)
# ------------------------------------------------------------------------------

def train_and_evaluate():
    """
    This function integrates training (Step 5), evaluation (Step 6),
    and final testing. It trains the agent for a number of episodes, performs PPO updates,
    and every fixed interval evaluates the agent and logs/plots the performance.
    
    The hyperparameters are scaled down for a lightweight experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create the environment.
    env = SimpleHUNLEnv(seed=42)

    # Initialize the PPO agent (from Steps 3-4).
    # The PPOAgent class is defined in the training code (see step 5).
    # For integration purposes, we define a simple PPOAgent class here.
    class PPOAgent:
        def __init__(self, device):
            self.device = device
            self.model = AlphaHoldemNet(
                num_actions=3,
                card_channels=7,
                card_height=4,
                card_width=13,
                history_max_len=6,
                history_num_actions=3,
                use_history=True
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        
        def select_action(self, card_tensor, history_tensor, deterministic=False):
            self.model.eval()
            with torch.no_grad():
                logits, value = self.model(card_tensor, history_tensor)
                probs = torch.softmax(logits, dim=-1)
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = torch.multinomial(probs, num_samples=1).squeeze(1)
                log_probs = torch.log(probs + 1e-8)
                chosen_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
            self.model.train()
            return action.item(), chosen_log_prob, value.squeeze(0)

    agent = PPOAgent(device)

    # Rollout buffer for storing episode data.
    class RolloutBuffer:
        def __init__(self):
            self.card_tensors = []
            self.history_tensors = []
            self.actions = []
            self.log_probs = []
            self.values = []
            self.rewards = []
            self.dones = []

        def clear(self):
            self.card_tensors = []
            self.history_tensors = []
            self.actions = []
            self.log_probs = []
            self.values = []
            self.rewards = []
            self.dones = []

        def add(self, card_tensor, history_tensor, action, log_prob, value, reward, done):
            self.card_tensors.append(card_tensor)
            self.history_tensors.append(history_tensor)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)

        def get_batches(self):
            batch = {
                "card_tensors": torch.cat(self.card_tensors, dim=0),
                "history_tensors": torch.cat(self.history_tensors, dim=0),
                "actions": torch.tensor(self.actions, dtype=torch.long, device=self.card_tensors[0].device),
                "old_log_probs": torch.stack(self.log_probs),
                "values": torch.stack(self.values).squeeze(),
                "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=self.card_tensors[0].device),
                "dones": torch.tensor(self.dones, dtype=torch.float32, device=self.card_tensors[0].device)
            }
            return batch

    buffer = RolloutBuffer()

    # Hyperparameters (scaled version)
    TOTAL_EPISODES = 5000         # Total episodes for training (hand count)
    ROLLOUT_SIZE = 64             # Update after 64 episodes collected
    PPO_EPOCHS = 3                # Number of PPO epochs per update
    MINI_BATCH_SIZE = 16          # Mini-batch size
    GAMMA = 0.99                  # Discount factor
    EPSILON = 0.2                 # PPO standard clipping ratio
    DELTA1 = 3.0                  # Additional clipping parameter
    DELTA2 = 10.0                 # Value loss lower clip bound
    DELTA3 = 10.0                 # Value loss upper clip bound
    ENTROPY_COEF = 0.01           # Entropy bonus coefficient
    VALUE_COEF = 0.5              # Value loss coefficient

    episode_rewards = []
    eval_avg_rewards = []
    eval_episodes = []
    
    total_steps = 0
    start_time = time.time()

    for episode in range(1, TOTAL_EPISODES + 1):
        obs = env.reset()
        # Convert observation to tensors.
        agent_hole = obs["hole_cards"].tolist()
        community_cards = obs["community_cards"].tolist()
        card_tensor = combine_card_channels(agent_hole, community_cards)
        card_tensor = card_tensor.unsqueeze(0).to(device)  # (1, 7, 4, 13)
        # For simplicity, we initialize an empty (zero) action history.
        history_tensor = torch.zeros((1, 6, 3), dtype=torch.float32, device=device)
        
        # Agent selects an action (stochastic during training).
        action, log_prob, value = agent.select_action(card_tensor, history_tensor, deterministic=False)
        _, reward, done, _ = env.step(action)
        total_steps += 1
        
        # Store transition in rollout buffer.
        buffer.add(card_tensor, history_tensor, action, log_prob, value, reward, done)
        episode_rewards.append(reward)
        
        # When rollout buffer is full, update the policy.
        if total_steps % ROLLOUT_SIZE == 0:
            batch = buffer.get_batches()
            buffer.clear()
            
            # Compute returns and advantage.
            returns = batch["rewards"]  # For one-step episodes, return = reward.
            advantages = returns - batch["values"]
            # Normalize advantages for stability.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Perform PPO update.
            batch_size = batch["actions"].shape[0]
            indices = np.arange(batch_size)
            
            for epoch in range(PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, batch_size, MINI_BATCH_SIZE):
                    end = start + MINI_BATCH_SIZE
                    mb_idx = indices[start:end]
                    mb_cards = batch["card_tensors"][mb_idx]
                    mb_history = batch["history_tensors"][mb_idx]
                    mb_actions = batch["actions"][mb_idx]
                    mb_old_log_probs = batch["old_log_probs"][mb_idx]
                    mb_values = batch["values"][mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    
                    # Forward pass.
                    logits, values = agent.model(mb_cards, mb_history)
                    policy_loss = compute_trinal_clip_policy_loss(
                        logits, mb_actions, mb_old_log_probs, mb_advantages,
                        epsilon=EPSILON, delta1=DELTA1
                    )
                    value_loss = compute_trinal_clip_value_loss(
                        values.squeeze(), mb_returns,
                        delta2=DELTA2, delta3=DELTA3
                    )
                    entropy_loss = compute_entropy_loss(logits)
                    total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss
                    
                    agent.optimizer.zero_grad()
                    total_loss.backward()
                    agent.optimizer.step()
            
            # Logging training reward for this rollout.
            rollout_avg = np.mean(episode_rewards[-ROLLOUT_SIZE:])
            print(f"Episode {episode}, Rollout Avg Reward: {rollout_avg:.4f}")
        
        # Evaluate every 500 episodes.
        if episode % 500 == 0:
            avg_eval_reward, _ = evaluate_agent(agent, env, num_episodes=200, device=device)
            eval_avg_rewards.append(avg_eval_reward)
            eval_episodes.append(episode)
            print(f"*** Evaluation after Episode {episode}: Avg Reward = {avg_eval_reward:.4f} ***")

    end_time = time.time()
    print("Training completed.")
    print(f"Total Episodes: {TOTAL_EPISODES}, Total Steps: {total_steps}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    # --------------------------------------------------------------------------
    # Step 6: Plotting Training and Evaluation Curves
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward over Training")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(eval_episodes, eval_avg_rewards, marker='o', label="Evaluation Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Evaluation Reward")
    plt.title("Evaluation Performance over Training")
    plt.legend()
    plt.show()

    # --------------------------------------------------------------------------
    # Step 7: Final Testing (Interactive Play)
    # --------------------------------------------------------------------------
    print("\n--- Final Interactive Testing ---")
    agent.model.eval()  # Set the model to evaluation mode.
    for i in range(10):
        obs = env.reset()
        agent_hole = obs["hole_cards"].tolist()
        community_cards = obs["community_cards"].tolist()
        card_tensor = combine_card_channels(agent_hole, community_cards)
        card_tensor = card_tensor.unsqueeze(0).to(device)
        history_tensor = torch.zeros((1, 6, 3), dtype=torch.float32, device=device)
        
        # Use deterministic (greedy) action selection during final testing.
        with torch.no_grad():
            logits, _ = agent.model(card_tensor, history_tensor)
            action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
        
        print(f"\nTest Hand {i+1}:")
        print("Agent Hole Cards:", agent_hole)
        print("Community Cards:", community_cards)
        print("Selected Action:", action)
        _, reward, done, info = env.step(action)
        print("Hand Reward:", reward)
        print("Info:", info)

if __name__ == '__main__':
    train_and_evaluate()
