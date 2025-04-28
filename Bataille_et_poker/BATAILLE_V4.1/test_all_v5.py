import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Global Game Parameters
# -----------------------------------------------------------
NB_CARTES = 52            # Deck of 52 unique cards
NB_CARTES_AGENT = 26      # Agent gets 26 cards
NB_TOURS = 26             # One card per round
SUITS = ["coeur", "trèfle", "carreau", "pique"]

# -----------------------------------------------------------
# Logging configuration (UTF-8)
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    encoding='utf-8')
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# -----------------------------------------------------------
# Card Encoding
# -----------------------------------------------------------
def encode_card(card):
    """
    Encode a card as a 5-dimensional vector:
      - First value: normalized rank (2 -> 0.0, Ace (14) -> 1.0).
      - Next 4 values: one-hot encoding of the suit in order: coeur, trèfle, carreau, pique.
    If card is None, returns a vector of zeros.
    """
    if card is None:
        return np.zeros(5, dtype=np.float32)
    rank, suit = card
    norm_rank = (rank - 2) / 12.0
    one_hot = np.zeros(4, dtype=np.float32)
    try:
        suit_index = SUITS.index(suit)
    except ValueError:
        suit_index = -1
    if suit_index >= 0:
        one_hot[suit_index] = 1.0
    return np.concatenate(([norm_rank], one_hot))

# -----------------------------------------------------------
# Environment: Bataille Game (Adapted)
# -----------------------------------------------------------
class BatailleEnv:
    def __init__(self):
        self.nb_tours = NB_TOURS
        self.reset()
    
    def reset(self):
        logging.debug("Resetting environment.")
        # Create deck: cards from rank 2 to 14 for each suit
        self.deck = [(rank, suit) for suit in SUITS for rank in range(2, 15)]
        random.shuffle(self.deck)
        # Deal: first half to adversary, second half to agent
        self.adversaire_main = self.deck[:NB_CARTES_AGENT]
        self.agent_main = self.deck[NB_CARTES_AGENT:]
        self.tour = 0
        # Adversary plays first
        self.nouvelle_manche()
        return self.get_observation()
    
    def nouvelle_manche(self):
        """Adversary plays a random card from its hand."""
        if len(self.adversaire_main) > 0:
            idx = random.randrange(len(self.adversaire_main))
            self.current_adversaire_carte = self.adversaire_main.pop(idx)
            logging.debug("Round %d: Adversary plays %s", self.tour, self.current_adversaire_carte)
        else:
            self.current_adversaire_carte = None

    def get_observation(self):
        """
        Returns a dictionary observation for the model:
          - 'hand': numpy array of shape (26, 5) representing the agent's hand.
          - 'adv': numpy array of shape (5,) representing the adversary's played card.
        """
        hand_obs = [encode_card(card) for card in self.agent_main]
        # Pad with zeros if hand has less than 26 cards
        while len(hand_obs) < NB_CARTES_AGENT:
            hand_obs.append(np.zeros(5, dtype=np.float32))
        hand_obs = np.stack(hand_obs)  # (26, 5)
        adv_obs = encode_card(self.current_adversaire_carte)  # (5,)
        return {'hand': hand_obs.astype(np.float32), 'adv': adv_obs.astype(np.float32)}
    
    def get_observation_dict(self):
        """
        Returns the raw observation (for rule-based and random agents):
          - 'agent_main': list of cards (tuples or None) for the agent.
          - 'adversaire_card': the card played by the adversary this round.
        """
        return {'agent_main': self.agent_main, 'adversaire_card': self.current_adversaire_carte}
    
    def get_valid_action_mask(self):
        """
        Returns a mask (length 26) indicating valid positions in the agent's hand.
          - Valid if the card is not None.
          - If the agent has at least one card matching the adversary's suit,
            only those positions are marked valid.
        """
        mask = np.zeros(NB_CARTES_AGENT, dtype=np.float32)
        for i in range(len(self.agent_main)):
            if self.agent_main[i] is not None:
                mask[i] = 1.0
        adv_suit = self.current_adversaire_carte[1] if self.current_adversaire_carte is not None else None
        possede_couleur = any(card is not None and card[1] == adv_suit for card in self.agent_main)
        if possede_couleur:
            for i in range(len(self.agent_main)):
                if self.agent_main[i] is not None and self.agent_main[i][1] != adv_suit:
                    mask[i] = 0.0
        logging.debug("Valid action mask: %s", mask)
        return mask
    
    def step(self, action):
        """
        The agent plays the card at the given index.
        Rules:
          - If the played card is of the same suit as the adversary:
              * If its rank is higher → reward = +1
              * Otherwise → reward = -1
          - Else (wrong suit) → reward = -1.
        The played card is removed from the agent's hand.
        """
        logging.debug("Agent action: %d", action)
        if action < 0 or action >= NB_CARTES_AGENT or self.agent_main[action] is None:
            reward = -1.0
            logging.debug("Invalid action: index out of range or no card.")
        else:
            carte_agent = self.agent_main[action]
            adv_carte = self.current_adversaire_carte
            logging.debug("Agent card: %s, Adversary card: %s", carte_agent, adv_carte)
            if carte_agent[1] == adv_carte[1]:
                reward = 1.0 if carte_agent[0] > adv_carte[0] else -1.0
            else:
                reward = -1.0
            self.agent_main[action] = None
        
        self.tour += 1
        done = (self.tour >= self.nb_tours)
        if not done:
            self.nouvelle_manche()
        # Return observation in two formats:
        # - For the model agent, use the encoded observation.
        # - For rule-based/random agents, they can call get_observation_dict() externally.
        return self.get_observation(), reward, done, {}
    
    def render(self):
        logging.info("Round: %d", self.tour)
        logging.info("Agent hand: %s", self.agent_main)
        logging.info("Adversary card: %s", self.current_adversaire_carte)

# -----------------------------------------------------------
# PPO Network (Two Inputs: hand and adversary card)
# -----------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # Process agent's hand: shape (26, 5)
        self.hand_fc = nn.Linear(5, hidden_dim)
        # Process adversary card: shape (5,)
        self.adv_fc = nn.Linear(5, hidden_dim)
        # Combine representations: hand gives 26*hidden_dim, adv gives hidden_dim
        self.fc1 = nn.Linear(26 * hidden_dim + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, hand, adv):
        # hand: (batch, 26, 5), adv: (batch, 5)
        hand_repr = F.relu(self.hand_fc(hand))  # (batch, 26, hidden_dim)
        hand_repr = hand_repr.view(hand_repr.size(0), -1)  # (batch, 26*hidden_dim)
        adv_repr = F.relu(self.adv_fc(adv))  # (batch, hidden_dim)
        x = torch.cat([hand_repr, adv_repr], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

# -----------------------------------------------------------
# Model Agent (PPO) that loads the best_model.pt
# -----------------------------------------------------------
class ModelAgent:
    def __init__(self, hidden_dim=128, model_path="best_model.pt"):
        self.model = ActorCriticNetwork(hidden_dim, NB_CARTES_AGENT)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def select_action(self, obs, valid_mask):
        """
        For a given observation (dictionary with 'hand' and 'adv') and valid action mask,
        returns the chosen action, log-probability and estimated value.
        """
        hand = torch.from_numpy(obs['hand']).float().unsqueeze(0)  # (1, 26, 5)
        adv = torch.from_numpy(obs['adv']).float().unsqueeze(0)    # (1, 5)
        logits, value = self.model(hand, adv)
        logits = logits.squeeze(0)
        # Mask invalid actions: add a very negative number so they are not chosen
        mask_tensor = torch.from_numpy(valid_mask).float()
        masked_logits = logits + (1 - mask_tensor) * (-1e8)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob, value.item()

# -----------------------------------------------------------
# Rule-Based Agent (uses the environment's internal state)
# -----------------------------------------------------------
class RuleBasedAgent:
    def select_action(self, observation):
        """
        Expects observation from get_observation_dict() containing:
          - 'agent_main': list of cards (tuples or None)
          - 'adversaire_card': the adversary's played card (tuple)
        Implements a basic strategy:
          - If agent has cards matching the adversary’s suit, play the lowest winning card if possible,
            else play the lowest card of that suit.
          - Otherwise, choose the suit with the most cards and play the lowest card from it.
        """
        hand = observation['agent_main']
        adv_card = observation['adversaire_card']
        if adv_card is None:
            return -1
        adv_rank, adv_suit = adv_card
        same_suit = [(i, card) for i, card in enumerate(hand) if card is not None and card[1] == adv_suit]
        if same_suit:
            winning = [(i, card) for i, card in same_suit if card[0] > adv_rank]
            if winning:
                chosen = min(winning, key=lambda x: x[1][0])
                return chosen[0]
            else:
                chosen = min(same_suit, key=lambda x: x[1][0])
                return chosen[0]
        else:
            suit_counts = {}
            for card in hand:
                if card is not None:
                    suit_counts[card[1]] = suit_counts.get(card[1], 0) + 1
            if not suit_counts:
                return -1
            best_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            candidates = [(i, card) for i, card in enumerate(hand) if card is not None and card[1] == best_suit]
            chosen = min(candidates, key=lambda x: x[1][0])
            return chosen[0]

# -----------------------------------------------------------
# Random Agent (selects a random valid action)
# -----------------------------------------------------------
class RandomAgent:
    def select_action(self, observation):
        """
        Expects observation from get_observation_dict() with key 'agent_main'.
        Selects a random valid index.
        """
        hand = observation['agent_main']
        valid_indices = [i for i, card in enumerate(hand) if card is not None]
        return random.choice(valid_indices) if valid_indices else -1

# -----------------------------------------------------------
# Simulation Function
# -----------------------------------------------------------
def simulate_games(agent, num_games=1000):
    rewards = []
    win_rates = []
    for _ in range(num_games):
        env = BatailleEnv()
        total_reward = 0
        wins = 0
        # For model agent, use encoded observation; for others, use raw dict.
        if isinstance(agent, ModelAgent):
            obs = env.get_observation()
        else:
            obs = env.get_observation_dict()
        for _ in range(NB_TOURS):
            if isinstance(agent, ModelAgent):
                valid_mask = env.get_valid_action_mask()
                action, _, _ = agent.select_action(obs, valid_mask)
                obs, reward, done, _ = env.step(action)
                # Continue using encoded obs for model agent.
            else:
                action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                # For rule-based/random, get updated raw observation.
                obs = env.get_observation_dict()
            total_reward += reward
            if reward > 0:
                wins += 1
            if done:
                break
        rewards.append(total_reward)
        win_rates.append(wins / NB_TOURS)
    return rewards, win_rates

# -----------------------------------------------------------
# Main Testing Loop and Plotting
# -----------------------------------------------------------
def main():
    num_games = 1000

    # Instantiate agents
    agent_rule = RuleBasedAgent()
    agent_random = RandomAgent()
    # Model agent using the new architecture – ensure best_model.pt is in the same folder.
    agent_model = ModelAgent(model_path="best_model.pt")

    logging.info("Simulating games with the Rule-Based Agent...")
    rewards_rule, win_rates_rule = simulate_games(agent_rule, num_games)
    logging.info("Simulating games with the Random Agent...")
    rewards_random, win_rates_random = simulate_games(agent_random, num_games)
    logging.info("Simulating games with the Model Agent...")
    rewards_model, win_rates_model = simulate_games(agent_model, num_games)
    
    # Plot rewards over games
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_rule, alpha=0.5, label="Rule-Based")
    plt.plot(np.convolve(rewards_rule, np.ones(100)/100, mode='valid'), color='red', label="Moving Average (Rule-Based)")
    plt.plot(rewards_random, alpha=0.5, label="Random")
    plt.plot(np.convolve(rewards_random, np.ones(100)/100, mode='valid'), color='blue', label="Moving Average (Random)")
    plt.plot(rewards_model, alpha=0.5, label="Model")
    plt.plot(np.convolve(rewards_model, np.ones(100)/100, mode='valid'), color='green', label="Moving Average (Model)")
    plt.title("Total Reward per Game")
    plt.xlabel("Game")
    plt.ylabel("Total Reward")
    plt.legend()
    
    # Histogram of rewards
    plt.subplot(1, 2, 2)
    plt.hist(rewards_rule, bins=30, alpha=0.7, label="Rule-Based", edgecolor='black')
    plt.hist(rewards_random, bins=30, alpha=0.7, label="Random", edgecolor='black')
    plt.hist(rewards_model, bins=30, alpha=0.7, label="Model", edgecolor='black')
    plt.title(f"Reward Distribution over {num_games} Games")
    plt.xlabel("Total Reward")
    plt.ylabel("Number of Games")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot win rates over games
    plt.figure(figsize=(7, 5))
    plt.plot(win_rates_rule, alpha=0.5, label="Rule-Based")
    plt.plot(np.convolve(win_rates_rule, np.ones(100)/100, mode='valid'), color='red', label="Moving Average (Rule-Based)")
    plt.plot(win_rates_random, alpha=0.5, label="Random")
    plt.plot(np.convolve(win_rates_random, np.ones(100)/100, mode='valid'), color='blue', label="Moving Average (Random)")
    plt.plot(win_rates_model, alpha=0.5, label="Model")
    plt.plot(np.convolve(win_rates_model, np.ones(100)/100, mode='valid'), color='green', label="Moving Average (Model)")
    plt.title("Win Rate per Game")
    plt.xlabel("Game")
    plt.ylabel("Win Rate (Fraction of Tricks Won)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
