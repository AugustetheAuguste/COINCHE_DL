import numpy as np
import torch
import torch.nn.functional as F

def one_hot_card(card):
    """
    Convert a single card (integer 0-51) into a one-hot encoded 2D tensor of shape (4, 13).
    Here:
      - rank = card // 4 (0 to 12)
      - suit = card % 4 (0 to 3)
    """
    tensor = torch.zeros((4, 13), dtype=torch.float32)
    rank = card // 4
    suit = card % 4
    tensor[suit, rank] = 1.0
    return tensor

def encode_cards(cards):
    """
    Given a list of card integers, encode each card as a one-hot 4x13 tensor,
    then stack them along a new channel dimension.
    Output shape: (N, 4, 13), where N is number of cards.
    """
    encoded = [one_hot_card(card) for card in cards]
    return torch.stack(encoded, dim=0)

def combine_card_channels(agent_hole, community_cards):
    """
    Combine the agent's hole cards and community cards into a single tensor.
    The resulting tensor will have shape (7, 4, 13) where:
      - Channels 0-1: agent's hole cards
      - Channels 2-6: community cards (5 cards)
    """
    hole_tensor = encode_cards(agent_hole)           # shape: (2, 4, 13)
    community_tensor = encode_cards(community_cards)   # shape: (5, 4, 13)
    combined = torch.cat([hole_tensor, community_tensor], dim=0)
    return combined  # shape: (7, 4, 13)

def encode_betting_history(bet_history, max_rounds=4, num_actions=6, output_channels=24):
    """
    Given a bet_history matrix of shape (max_rounds, 2), where each row contains the discrete action
    taken by the agent and its opponent in that betting round, encode it as a rich betting history tensor.
    
    We design a simple scheme as follows:
      1. Compute one-hot encodings for the agent and opponent:
           - agent_onehot: shape (max_rounds, num_actions)
           - opp_onehot: shape (max_rounds, num_actions)
      2. Compute a combined one-hot as the element-wise maximum of the two.
      3. Tile each of these three matrices eight times along a new channel dimension
         and then concatenate them to form a tensor of shape (24, max_rounds, num_actions).
    
    Parameters:
      bet_history: np.array of shape (max_rounds, 2) with discrete action indices (0 to num_actions-1).
      max_rounds: Maximum number of betting rounds (default 4, for Pre-flop, Flop, Turn, River).
      num_actions: Total number of discrete actions (for example, 6).
      output_channels: Total output channels (we design for 24 channels).
      
    Returns:
      A torch tensor of shape (output_channels, max_rounds, num_actions) of type float32.
    """
    # Initialize one-hot matrices for agent and opponent.
    # For each round, convert the action to a one-hot vector.
    agent_onehot = np.zeros((max_rounds, num_actions), dtype=np.float32)
    opp_onehot = np.zeros((max_rounds, num_actions), dtype=np.float32)
    
    for i in range(max_rounds):
        # Assume bet_history[i, :] contains valid actions.
        # If a round was not played, the value might be 0.
        agent_action = int(bet_history[i, 0])
        opp_action = int(bet_history[i, 1])
        agent_onehot[i, agent_action] = 1.0
        opp_onehot[i, opp_action] = 1.0

    # Combined one-hot: elementwise maximum (logical OR).
    combined = np.maximum(agent_onehot, opp_onehot)

    # Convert to torch tensors.
    agent_onehot = torch.tensor(agent_onehot)   # shape: (max_rounds, num_actions)
    opp_onehot = torch.tensor(opp_onehot)
    combined = torch.tensor(combined)

    # Determine the number of repeats for each group.
    # We want total channels = 24, and we have 3 groups; so each group gets 8 channels.
    repeats = output_channels // 3  # e.g., 8

    agent_rep = agent_onehot.unsqueeze(0).repeat(repeats, 1, 1)  # shape: (8, max_rounds, num_actions)
    opp_rep = opp_onehot.unsqueeze(0).repeat(repeats, 1, 1)      # shape: (8, max_rounds, num_actions)
    comb_rep = combined.unsqueeze(0).repeat(repeats, 1, 1)        # shape: (8, max_rounds, num_actions)

    # Concatenate along the channel dimension.
    rich_history = torch.cat([agent_rep, opp_rep, comb_rep], dim=0)  # shape: (24, max_rounds, num_actions)
    return rich_history

# ------------------------------
# Testing the Rich Betting History Encoding
# ------------------------------
if __name__ == '__main__':
    # Test the card encoding functions.
    agent_hole = [10, 25]
    community_cards = [3, 17, 30, 44, 51]
    card_tensor = combine_card_channels(agent_hole, community_cards)
    print("Combined Card Tensor Shape (should be 7x4x13):", card_tensor.shape)

    # Test the betting history encoder.
    # Let's simulate a bet_history for 4 rounds.
    # Each round has two actions (agent and opponent).
    # For example, suppose:
    # Round 0: agent: raise with option index 3, opponent: call (action 1)
    # Round 1: agent: call (1), opponent: raise (option 4)
    # Round 2: agent: check/call (1), opponent: check/call (1)
    # Round 3: agent: raise (option 2), opponent: fold (0)
    bet_history = np.array([
        [3, 1],
        [1, 4],
        [1, 1],
        [2, 0]
    ], dtype=np.int32)
    rich_history = encode_betting_history(bet_history, max_rounds=4, num_actions=6, output_channels=24)
    print("Rich Betting History Tensor Shape (should be 24x4x6):", rich_history.shape)
    print("Rich Betting History Tensor:\n", rich_history)
