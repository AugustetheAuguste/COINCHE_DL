import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Updated Network Architecture: AlphaHoldemNetExpanded
###############################################################################
class AlphaHoldemNetExpanded(nn.Module):
    """
    Updated network architecture for a realistic HUNL agent.
    
    This network accepts two inputs:
      - A card tensor of shape (batch, 7, 4, 13) representing the agent's hole cards and community cards.
      - A rich betting history tensor of shape (batch, 24, 4, 6) representing the detailed betting history.
      
    The network processes these two inputs via separate convolutional branches and fuses their flattened outputs.
    It then outputs:
      - An actor head: logits for a larger action space (6 discrete actions)
      - A critic head: a scalar value estimate.
    """
    def __init__(self, num_actions=6):
        super(AlphaHoldemNetExpanded, self).__init__()
        
        # ----- Card Branch -----
        # Input shape: (batch, 7, 4, 13)
        self.card_conv = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, padding=1),  # -> (32, 4, 13)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # -> (64, 4, 13)
            nn.ReLU(),
            nn.Flatten()  # -> (64 * 4 * 13)
        )
        card_feature_size = 64 * 4 * 13  # 64*52 = 3328

        # ----- Betting History Branch -----
        # Input shape: (batch, 24, 4, 6)
        self.bet_conv = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, padding=1),  # -> (32, 4, 6)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # -> (64, 4, 6)
            nn.ReLU(),
            nn.Flatten()  # -> (64 * 4 * 6)
        )
        bet_feature_size = 64 * 4 * 6  # 64*24 = 1536

        # ----- Fusion -----
        combined_feature_size = card_feature_size + bet_feature_size
        self.fusion_fc = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU()
        )

        # ----- Actor Head -----
        self.actor_head = nn.Linear(256, num_actions)

        # ----- Critic Head -----
        self.critic_head = nn.Linear(256, 1)

    def forward(self, card_tensor, bet_history_tensor):
        """
        Forward pass.
        
        Parameters:
          card_tensor: Tensor of shape (batch, 7, 4, 13)
          bet_history_tensor: Tensor of shape (batch, 24, 4, 6)
        
        Returns:
          logits: Tensor of shape (batch, num_actions)
          value: Tensor of shape (batch,) representing the state value.
        """
        # Process card branch.
        card_features = self.card_conv(card_tensor)  # shape: (batch, card_feature_size)
        
        # Process betting history branch.
        bet_features = self.bet_conv(bet_history_tensor)  # shape: (batch, bet_feature_size)
        
        # Concatenate features from both branches.
        combined_features = torch.cat([card_features, bet_features], dim=1)  # shape: (batch, combined_feature_size)
        fusion_features = self.fusion_fc(combined_features)  # shape: (batch, 256)
        
        # Actor head outputs logits.
        logits = self.actor_head(fusion_features)  # shape: (batch, num_actions)
        # Critic head outputs a scalar value.
        value = self.critic_head(fusion_features).squeeze(1)  # shape: (batch,)
        
        return logits, value

###############################################################################
# PPO with Trinal-Clip Loss Functions
###############################################################################
def compute_trinal_clip_policy_loss(new_logits, actions, old_log_probs, advantages, epsilon=0.2, delta1=3.0):
    """
    Compute PPO policy loss with an extra clipping for negative advantages.
    
    Parameters:
      new_logits: Tensor of shape (batch, num_actions) from current policy.
      actions: Tensor of shape (batch,) with chosen actions.
      old_log_probs: Tensor of shape (batch,) with log probabilities from the old policy.
      advantages: Tensor of shape (batch,) with advantage estimates.
      epsilon: Standard PPO clip ratio (default 0.2).
      delta1: Additional clipping hyper-parameter for negative advantages (must be > 1+epsilon).
    
    Returns:
      policy_loss: A scalar tensor representing the policy loss.
    """
    # Calculate new log probabilities.
    new_log_probs_all = F.log_softmax(new_logits, dim=-1)  # shape: (batch, num_actions)
    new_log_probs = new_log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape: (batch,)
    
    # Compute probability ratio.
    r_t = torch.exp(new_log_probs - old_log_probs)  # shape: (batch,)
    
    # Standard PPO clipping.
    clipped_r = torch.clamp(r_t, 1 - epsilon, 1 + epsilon)
    
    # Additional clipping for negative advantages.
    trinal_r = torch.where(advantages < 0, torch.clamp(r_t, 1 - epsilon, delta1), clipped_r)
    
    # Surrogate loss.
    surrogate_loss = torch.min(r_t * advantages, trinal_r * advantages)
    policy_loss = -torch.mean(surrogate_loss)
    return policy_loss

def compute_trinal_clip_value_loss(values, returns, delta2=10.0, delta3=10.0):
    """
    Compute value function loss with clipping.
    
    Parameters:
      values: Tensor of shape (batch,) – current value estimates.
      returns: Tensor of shape (batch,) – target returns.
      delta2: Lower clip bound (absolute value).
      delta3: Upper clip bound (absolute value).
    
    Returns:
      value_loss: A scalar tensor representing the MSE loss between values and clipped returns.
    """
    clipped_returns = torch.clamp(returns, -delta2, delta3)
    value_loss = F.mse_loss(values, clipped_returns)
    return value_loss

def compute_entropy_loss(logits):
    """
    Compute entropy loss for encouraging exploration.
    
    Parameters:
      logits: Tensor of shape (batch, num_actions)
    
    Returns:
      entropy: A scalar tensor of the entropy loss.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))
    return entropy

###############################################################################
# Testing the Expanded Network and PPO Losses
###############################################################################
if __name__ == '__main__':
    batch_size = 2
    
    # Create dummy card tensor: shape (batch_size, 7, 4, 13)
    card_tensor = torch.randint(0, 2, (batch_size, 7, 4, 13)).float()
    
    # Create dummy betting history tensor: shape (batch_size, 24, 4, 6)
    bet_history_tensor = torch.randint(0, 2, (batch_size, 24, 4, 6)).float()
    
    # Instantiate the network.
    model = AlphaHoldemNetExpanded(num_actions=6)
    
    # Forward pass.
    logits, values = model(card_tensor, bet_history_tensor)
    print("=== Network Forward Pass ===")
    print("Logits shape:", logits.shape)    # Expect: (batch_size, 6)
    print("Values shape:", values.shape)      # Expect: (batch_size,)
    
    # Test PPO loss functions.
    actions = torch.randint(0, 6, (batch_size,))  # Random actions in expanded space
    old_log_probs = torch.zeros(batch_size)       # For simplicity
    advantages = torch.randn(batch_size)          # Random advantage values
    returns = torch.randn(batch_size)             # Random returns
    
    policy_loss = compute_trinal_clip_policy_loss(logits, actions, old_log_probs, advantages, epsilon=0.2, delta1=3.0)
    value_loss = compute_trinal_clip_value_loss(values, returns, delta2=10.0, delta3=10.0)
    entropy_loss = compute_entropy_loss(logits)
    
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    
    print("\n=== Losses ===")
    print("Policy Loss: {:.4f}".format(policy_loss.item()))
    print("Value Loss: {:.4f}".format(value_loss.item()))
    print("Entropy Loss: {:.4f}".format(entropy_loss.item()))
    print("Total Loss: {:.4f}".format(total_loss.item()))
