import gym
from gym import spaces
import numpy as np
import random

def naive_hand_strength(cards):
    """
    A naive hand strength evaluator.
    Sums the rank values (card // 4) and adds bonus for pairs/trips.
    """
    ranks = [card // 4 for card in cards]
    score = sum(ranks)
    for rank in set(ranks):
        count = ranks.count(rank)
        if count == 2:
            score += 5   # bonus for pair
        elif count == 3:
            score += 10  # bonus for trips
        elif count == 4:
            score += 20  # bonus for quads
    return score

class RealisticHUNLEnvLegal(gym.Env):
    """
    A heads-up no-limit Texas Hold'em environment with multi-step betting rounds,
    legal action constraints, and an expanded discrete action space.
    
    Game Phases:
      1. Pre-flop:
         - Forced blinds: agent posts small blind; opponent posts big blind.
         - Deal two hole cards per player.
         - Betting round until contributions match.
      2. Flop:
         - Deal three community cards.
         - Betting round.
      3. Turn:
         - Deal one community card.
         - Betting round.
      4. River:
         - Deal one community card.
         - Betting round.
      5. Showdown:
         - Compare hands using a naive evaluator.
    
    Expanded Action Space:
      Actions are defined as:
        0: Fold  
        1: Check/Call  
        2: Raise with multiplier 1  
        3: Raise with multiplier 2  
        4: Raise with multiplier 4  
        5: Raise with multiplier 8  
      
      Legal Action Mask:
        At each decision, the environment computes a binary mask over actions.
      
    Observations:
      A dictionary with:
        - "hole_cards": agent’s private cards (array of 2 ints)
        - "community_cards": array of 5 ints (not-dealt cards represented as 52)
        - "pot": current pot size (int)
        - "round": round index (0: Pre-flop, 1: Flop, 2: Turn, 3: River, 4: Showdown)
        - "bet_history": matrix (shape (4,2)) of actions for each round
        - "legal_actions": binary vector of length equal to action_space.n.
    """
    def __init__(self, seed=None, raise_options=[1, 2, 4, 8]):
        super(RealisticHUNLEnvLegal, self).__init__()
        self.seed(seed)
        self.full_deck = list(range(52))
        self.round_names = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
        self.current_round = 0

        # Blinds and base raise
        self.small_blind = 1
        self.big_blind = 2
        self.base_raise = 2  # base raise amount
        self.last_raise = self.base_raise  # store last raise amount

        # Raise options multipliers (e.g., [1, 2, 4, 8])
        self.raise_options = raise_options

        # Expanded action space: actions: 0: Fold, 1: Check/Call, 2-5: Raise options.
        self.num_actions = 2 + len(self.raise_options)
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space includes legal actions mask.
        self.observation_space = spaces.Dict({
            "hole_cards": spaces.MultiDiscrete([52, 52]),
            "community_cards": spaces.MultiDiscrete([53]*5),
            "pot": spaces.Discrete(1000),
            "round": spaces.Discrete(len(self.round_names)),
            "bet_history": spaces.Box(low=0, high=self.num_actions-1, shape=(4, 2), dtype=np.int32),
            "legal_actions": spaces.MultiBinary(self.num_actions)
        })

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _initialize_deck(self):
        deck = self.full_deck.copy()
        random.shuffle(deck)
        return deck

    def _deal_private_cards(self):
        self.agent_hole = [self.deck.pop(), self.deck.pop()]
        self.opponent_hole = [self.deck.pop(), self.deck.pop()]

    def _deal_community_cards(self, num):
        return [self.deck.pop() for _ in range(num)]

    def reset(self):
        """ Reset the hand. """
        self.deck = self._initialize_deck()
        self.current_round = 0
        self._deal_private_cards()
        self.community_cards = [-1] * 5
        # Pre-flop: blinds are posted.
        self.pot = self.small_blind + self.big_blind
        self.contributions = {"agent": self.small_blind, "opponent": self.big_blind}
        self.last_raise = self.base_raise

        self.bet_history = np.zeros((4, 2), dtype=np.int32)
        self.current_bet = self.big_blind
        self.to_act = "agent"
        self.betting_round_over = False
        self.done = False

        return self._get_observation()

    def _get_observation(self):
        """ Build observation including legal actions mask. """
        comm_cards = [card if card != -1 else 52 for card in self.community_cards]
        legal_mask = self._compute_legal_actions()
        obs = {
            "hole_cards": np.array(self.agent_hole, dtype=np.int32),
            "community_cards": np.array(comm_cards, dtype=np.int32),
            "pot": self.pot,
            "round": self.current_round,
            "bet_history": self.bet_history.copy(),
            "legal_actions": legal_mask
        }
        return obs

    def _compute_legal_actions(self):
        """
        Compute legal actions based on the current betting state.
        For simplicity, we assume all actions are legal.
        """
        legal = np.ones(self.num_actions, dtype=np.int32)
        return legal

    def _advance_round(self):
        """ Advance to the next round, resetting betting state. """
        self.current_round += 1
        self.betting_round_over = False
        self.current_bet = 0
        self.contributions = {"agent": 0, "opponent": 0}
        self.last_raise = self.base_raise

        if self.current_round == 1:  # Flop
            flop = self._deal_community_cards(3)
            for i in range(3):
                self.community_cards[i] = flop[i]
        elif self.current_round == 2:  # Turn
            turn = self._deal_community_cards(1)
            self.community_cards[3] = turn[0]
        elif self.current_round == 3:  # River
            river = self._deal_community_cards(1)
            self.community_cards[4] = river[0]
        elif self.current_round == 4:  # Showdown
            self.done = True

        self.to_act = "agent"

    def _is_betting_settled(self):
        """ Check if current round betting is settled (equal contributions). """
        return self.contributions["agent"] == self.contributions["opponent"]

    def step(self, action):
        """
        Process one player's action in the betting round.
        Expanded action space:
          0: Fold
          1: Check/Call
          2-...: Raise (using raise_options multipliers)
          
        Returns:
          observation, reward, done, info
        """
        if self.done:
            raise Exception("Hand is over. Please call reset().")
        info = {}
        reward = 0
        current_player = self.to_act

        # Check legal action mask.
        legal_mask = self._compute_legal_actions()
        if legal_mask[action] == 0:
            reward = -10  # Penalty for illegal action.
            self.done = True
            info["result"] = f"{current_player}_illegal_action"
            return self._get_observation(), reward, self.done, info

        if current_player == "agent":
            self.bet_history[self.current_round, 0] = action
            if action == 0:  # Fold
                reward = -self.pot
                self.done = True
                info["result"] = "agent_folded"
                return self._get_observation(), reward, self.done, info
            elif action == 1:  # Check/Call
                required = self.current_bet - self.contributions["agent"]
                self.contributions["agent"] += required
                self.pot += required
            else:  # Raise
                multiplier = self.raise_options[action - 2]
                required_call = max(0, self.current_bet - self.contributions["agent"])
                raise_amt = multiplier * max(self.base_raise, self.last_raise)
                total = required_call + raise_amt
                self.contributions["agent"] += total
                self.pot += total
                self.current_bet = self.contributions["agent"]
                self.last_raise = raise_amt

            self.to_act = "opponent"
            return self._get_observation(), 0, self.done, info

        else:
            # Opponent's turn.
            # Use a simple heuristic: if call required is low, then call; otherwise, choose among call, raise, or fold.
            opp_required = self.current_bet - self.contributions["opponent"]
            if opp_required <= 0:
                opp_action = 1  # Check.
            else:
                opp_action = np.random.choice([1, 2, 3, 4, 5, 0], p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
            self.bet_history[self.current_round, 1] = opp_action

            if opp_action == 0:  # Fold
                reward = self.pot
                self.done = True
                info["result"] = "opponent_folded"
                self.to_act = "agent"
                return self._get_observation(), reward, self.done, info
            elif opp_action == 1:  # Check/Call
                required = self.current_bet - self.contributions["opponent"]
                self.contributions["opponent"] += required
                self.pot += required
            else:  # Raise
                multiplier = self.raise_options[opp_action - 2]
                required_call = max(0, self.current_bet - self.contributions["opponent"])
                raise_amt = multiplier * max(self.base_raise, self.last_raise)
                total = required_call + raise_amt
                self.contributions["opponent"] += total
                self.pot += total
                self.current_bet = self.contributions["opponent"]
                self.last_raise = raise_amt

            self.to_act = "agent"

            if self._is_betting_settled():
                self.betting_round_over = True
                if self.current_round < 4:
                    self._advance_round()
                    return self._get_observation(), 0, self.done, info
                else:
                    self.done = True
                    agent_hand = self.agent_hole + [card for card in self.community_cards if card != -1]
                    opp_hand = self.opponent_hole + [card for card in self.community_cards if card != -1]
                    agent_strength = naive_hand_strength(agent_hand)
                    opp_strength = naive_hand_strength(opp_hand)
                    info["agent_strength"] = agent_strength
                    info["opponent_strength"] = opp_strength
                    if agent_strength > opp_strength:
                        reward = self.pot
                        info["result"] = "agent_wins_showdown"
                    elif agent_strength < opp_strength:
                        reward = -self.pot
                        info["result"] = "agent_loses_showdown"
                    else:
                        reward = 0
                        info["result"] = "tie_showdown"
                    return self._get_observation(), reward, self.done, info

            return self._get_observation(), 0, self.done, info

    def render(self, mode="human"):
        print("Round:", self.round_names[self.current_round])
        print("Agent Hole Cards:", self.agent_hole)
        print("Community Cards:", self.community_cards)
        print("Pot:", self.pot)
        print("Contributions:", self.contributions)
        print("Current Bet to Call:", self.current_bet)
        print("Last Raise Amount:", self.last_raise)
        print("Bet History (Rounds 0-3):")
        print(self.bet_history)
        print("Next to Act:", self.to_act)
        legal = self._compute_legal_actions()
        print("Legal Actions Mask:", legal)
        print("-" * 40)

    def close(self):
        pass

# Testing the modified environment.
if __name__ == '__main__':
    env = RealisticHUNLEnvLegal(seed=42)
    obs = env.reset()
    print("Initial Observation:")
    print(obs)
    env.render()
    
    done = False
    while not done:
        legal = obs["legal_actions"]
        legal_actions = [i for i, allowed in enumerate(legal) if allowed == 1]
        action = random.choice(legal_actions)
        print("\nAgent chooses action:", action, 
              "(0: Fold, 1: Check/Call, 2-5: Raise with different multipliers)")
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Hand ended with reward:", reward, "Info:", info)
