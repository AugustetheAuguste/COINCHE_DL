# Import modules from previous steps.
# Ensure the following files are in your working directory:
#   - realistic_poker_env_legal.py (defines RealisticHUNLEnvLegal)
#   - state_representation.py (defines combine_card_channels, encode_betting_history)
#   - model_and_ppo_expanded.py (defines AlphaHoldemNetExpanded and PPO loss functions)
import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict


#########################################
# Hyperparameters
#########################################
TOTAL_EPISODES = 100_000_000
ROLLOUT_EPISODES = 64
PPO_EPOCHS = 6
MINI_BATCH_SIZE = 16
GAMMA = 0.99
LEARNING_RATE = 0.0003
EPSILON = 0.2
DELTA1 = 3.0
DELTA2 = 10.0
DELTA3 = 10.0
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

CARD_CHANNELS = 7
CARD_HEIGHT = 4
CARD_WIDTH = 13
BET_HISTORY_ROUNDS = 4
BET_HISTORY_OUTPUT_CHANNELS = 24
BET_HISTORY_NUM_ACTIONS = 6

PLOT_FREQ = 100

#########################################
# Helper function to compute discounted returns
#########################################
def compute_returns(rewards, masks, gamma):
    """
    Compute discounted returns for an episode.
    rewards: list of float (one per step)
    masks: list of 1 or 0 (1 if not terminal, 0 if terminal)
    """
    returns = []
    R = 0
    for r, m in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * m
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

#########################################
# PPOAgent using the expanded network
#########################################
class PPOAgent:
    def __init__(self, device, net_class, num_actions=6):
        self.device = device
        self.model = net_class(num_actions=num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    
    def select_action(self, card_tensor, bet_history_tensor, deterministic=False):
        """
        card_tensor: (1,7,4,13)
        bet_history_tensor: (1,24,4,6)
        """
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(card_tensor, bet_history_tensor)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, num_samples=1).squeeze(1)
            log_probs = torch.log(probs + 1e-8)
            chosen_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        self.model.train()
        return action.item(), chosen_log_prob, value.squeeze(0)

#########################################
# RolloutBuffer for full-hand episodes
#########################################
class RolloutBuffer:
    def __init__(self):
        self.states_cards = []
        self.states_bet_hist = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def clear(self):
        self.states_cards.clear()
        self.states_bet_hist.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()

#########################################
# Preflop folding probability matrix
#########################################
rank_names = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
def rank_idx(r):
    """Map rank in [0..12] (0=2..12=A) to a matrix index [0..12] (0->A,12->2)."""
    return 12 - r

def plot_preflop_folding_matrix(preflop_fold_by_rank, episode=0, savepath="plots/"):
    """
    Build a 13x13 matrix of fold probabilities for hole-card ranks.
    row=rank1, col=rank2, labeled by rank_names (A..2).
    """
    matrix = np.zeros((13,13), dtype=np.float32)
    for (r1, r2), (fold_count, total_count) in preflop_fold_by_rank.items():
        if total_count > 0:
            p = fold_count / total_count
        else:
            p = 0.0
        i = rank_idx(r1)
        j = rank_idx(r2)
        matrix[i,j] = p
        matrix[j,i] = p

    plt.figure(figsize=(6,5))
    plt.title(f"Preflop Folding Probability (Episode={episode})")
    cax = plt.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(cax, label="Fold Probability")
    plt.xticks(np.arange(13), rank_names, rotation=90)
    plt.yticks(np.arange(13), rank_names)
    plt.tight_layout()
    plt.savefig(f"{savepath}/preflop_fold_matrix_{episode}.png")
    plt.close()

#########################################
# Training Loop
#########################################
def train_full_episodes():
    import math
    from env import RealisticHUNLEnvLegal
    from state_representation import combine_card_channels, encode_betting_history
    from model_and_ppo import (
        AlphaHoldemNetExpanded,
        compute_trinal_clip_policy_loss,
        compute_trinal_clip_value_loss,
        compute_entropy_loss
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = RealisticHUNLEnvLegal(seed=42)
    agent = PPOAgent(device, AlphaHoldemNetExpanded, num_actions=BET_HISTORY_NUM_ACTIONS)
    buffer = RolloutBuffer()

    episode_rewards = []
    action_counter = np.zeros(BET_HISTORY_NUM_ACTIONS, dtype=np.int32)
    preflop_fold_by_rank = defaultdict(lambda: [0, 0])
    rewards_over_time = []
    loss_history = {"policy": [], "value": [], "entropy": []}

    total_steps = 0
    start_time = time.time()

    for episode in range(1, TOTAL_EPISODES + 1):
        obs = env.reset()
        done = False

        ep_states_cards = []
        ep_states_bet_hist = []
        ep_actions = []
        ep_log_probs = []
        ep_values = []
        ep_rewards = []
        ep_masks = []

        # For logging preflop folding:
        preflop_hole = obs["hole_cards"].tolist()

        while not done:
            # Convert obs to card_tensor
            agent_hole = obs["hole_cards"].tolist()
            community_cards = [c if c != 52 else -1 for c in obs["community_cards"].tolist()]
            card_tensor = combine_card_channels(agent_hole, community_cards).unsqueeze(0).to(device)

            # Convert bet_history to rich tensor
            bet_hist_np = obs["bet_history"]
            bet_hist_tensor = encode_betting_history(
                bet_hist_np,
                max_rounds=BET_HISTORY_ROUNDS,
                num_actions=BET_HISTORY_NUM_ACTIONS,
                output_channels=BET_HISTORY_OUTPUT_CHANNELS
            ).unsqueeze(0).to(device)

            # Record state
            ep_states_cards.append(card_tensor)
            ep_states_bet_hist.append(bet_hist_tensor)

            # Select action
            action, log_prob, value = agent.select_action(card_tensor, bet_hist_tensor, deterministic=False)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_values.append(value)

            # Preflop stats
            if obs["round"] == 0:
                action_counter[action] += 1
                # hole_ranks in ascending order
                sorted_ranks = tuple(sorted([r//4 for r in preflop_hole]))
                preflop_fold_by_rank[sorted_ranks][1] += 1
                if action == 0:
                    preflop_fold_by_rank[sorted_ranks][0] += 1

            obs, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            ep_masks.append(0 if done else 1)
            total_steps += 1

        # End of episode: compute returns
        returns = compute_returns(ep_rewards, ep_masks, GAMMA).to(device)

        # If we only have one step, ensure shape is (1,)
        ep_values_tensor = torch.stack(ep_values)
        if ep_values_tensor.dim() == 0:
            ep_values_tensor = ep_values_tensor.unsqueeze(0)
        if ep_values_tensor.numel() == 1 and returns.numel() == 1:
            # Both single-step
            ep_values_tensor = ep_values_tensor.view(1)
            returns = returns.view(1)

        # Compute advantages
        advantages = returns - ep_values_tensor.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Final reward
        episode_rewards.append(returns[0].item())
        rewards_over_time.append(returns[0].item())

        # Add transitions to buffer
        for i in range(len(ep_actions)):
            buffer.states_cards.append(ep_states_cards[i])
            buffer.states_bet_hist.append(ep_states_bet_hist[i])
            buffer.actions.append(ep_actions[i])
            buffer.log_probs.append(ep_log_probs[i])

            # If single-step, use item() to avoid dimension mismatch
            if len(ep_actions) == 1:
                buffer.values.append(ep_values_tensor[0].item())
            else:
                buffer.values.append(ep_values_tensor[i].item())

            buffer.rewards.append(returns[i].item())
            buffer.masks.append(ep_masks[i])

        # PPO update after ROLLOUT_EPISODES
        if episode % ROLLOUT_EPISODES == 0:
            # cat the rollout
            states_cards_batch = torch.cat(buffer.states_cards, dim=0).to(device)  # (N,7,4,13)
            states_bet_hist_batch = torch.cat(buffer.states_bet_hist, dim=0).to(device)  # (N,24,4,6)
            actions_batch = torch.tensor(buffer.actions, dtype=torch.long, device=device)
            old_log_probs_batch = torch.stack(buffer.log_probs).to(device)
            values_batch = torch.tensor(buffer.values, dtype=torch.float32, device=device)
            returns_batch = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
            masks_batch = torch.tensor(buffer.masks, dtype=torch.float32, device=device)

            advantages_batch = returns_batch - values_batch
            if advantages_batch.numel() > 1:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            dataset_size = actions_batch.shape[0]
            indices = np.arange(dataset_size)
            # PPO epochs
            for _ in range(PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, MINI_BATCH_SIZE):
                    end = start + MINI_BATCH_SIZE
                    mb_idx = indices[start:end]
                    mb_cards = states_cards_batch[mb_idx]
                    mb_bet_hist = states_bet_hist_batch[mb_idx]
                    mb_actions = actions_batch[mb_idx]
                    mb_old_log_probs = old_log_probs_batch[mb_idx]
                    mb_values = values_batch[mb_idx]
                    mb_returns = returns_batch[mb_idx]
                    mb_advantages = advantages_batch[mb_idx]

                    logits, val_out = agent.model(mb_cards, mb_bet_hist)
                    policy_loss = compute_trinal_clip_policy_loss(
                        logits, mb_actions, mb_old_log_probs, mb_advantages,
                        epsilon=EPSILON, delta1=DELTA1
                    )
                    value_loss = compute_trinal_clip_value_loss(
                        val_out, mb_returns,
                        delta2=DELTA2, delta3=DELTA3
                    )
                    entropy_loss = compute_entropy_loss(logits)

                    total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss
                    agent.optimizer.zero_grad()
                    total_loss.backward()
                    agent.optimizer.step()

                    loss_history["policy"].append(policy_loss.item())
                    loss_history["value"].append(value_loss.item())
                    loss_history["entropy"].append(entropy_loss.item())

            buffer.clear()
            avg_rollout_reward = np.mean(rewards_over_time[-ROLLOUT_EPISODES:])
            print(f"[Episode {episode}] RolloutAvgReward: {avg_rollout_reward:.4f}")

            # Save plots
            if episode % PLOT_FREQ == 0:
                if not os.path.exists("plots"):
                    os.makedirs("plots")
                
                # Episode rewards
                plt.figure(figsize=(8,4))
                plt.plot(rewards_over_time, label="Episode Reward")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title(f"Episode Reward up to Episode {episode}")
                plt.legend()
                plt.savefig(f"plots/episode_rewards_{episode}.png")
                plt.close()

                # Loss curves
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.plot(loss_history["policy"], label="Policy Loss")
                plt.xlabel("Mini-batch updates")
                plt.ylabel("Loss")
                plt.title("Policy Loss")
                plt.legend()

                plt.subplot(1,3,2)
                plt.plot(loss_history["value"], label="Value Loss", color="orange")
                plt.xlabel("Mini-batch updates")
                plt.ylabel("Loss")
                plt.title("Value Loss")
                plt.legend()

                plt.subplot(1,3,3)
                plt.plot(loss_history["entropy"], label="Entropy Loss", color="green")
                plt.xlabel("Mini-batch updates")
                plt.ylabel("Loss")
                plt.title("Entropy Loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"plots/losses_{episode}.png")
                plt.close()

                # Action distribution
                plt.figure(figsize=(8,4))
                actions_range = np.arange(BET_HISTORY_NUM_ACTIONS)
                plt.bar(actions_range, action_counter, tick_label=[str(a) for a in actions_range])
                plt.xlabel("Action")
                plt.ylabel("Frequency")
                plt.title(f"Agent Action Distribution up to Episode {episode}")
                plt.savefig(f"plots/action_distribution_{episode}.png")
                plt.close()

                # Plot preflop folding probability matrix
                plot_preflop_folding_matrix(preflop_fold_by_rank, episode=episode, savepath="plots")
                print(f"Saved plots for episode {episode}")

    end_time = time.time()
    print("Training completed.")
    print(f"Total Episodes: {TOTAL_EPISODES}, Total Steps: {total_steps}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    # Optional final evaluation
    # avg_eval_reward, _ = evaluate_agent(agent, env, num_episodes=200, device=device)
    # print(f"Final Evaluation Avg Reward: {avg_eval_reward:.4f}")


if __name__ == "__main__":
    # For safety, create a "plots" dir if needed
    if not os.path.exists("plots"):
        os.makedirs("plots")
    train_full_episodes()
