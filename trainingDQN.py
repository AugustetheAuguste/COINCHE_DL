from DQNannonce import DQNAgent, DQNannonce
from Environnement.annonce_suit import Annonce
from Modele.modele_jeu_IV import CoincheEnv
from Game_to_Train_DQN import MainCoincheAI

agent_annonce = DQNAgent(state_size=15, action_size=10, model=DQNannonce(15, 10))
env_annonce = CoincheEnv()

for episode in range(1000):
    state = env_annonce.reset()
    done = False
    while not done:
        action = agent_annonce.act(state)
        next_state, reward, done, _ = env_annonce.step(action)
        agent_annonce.remember(state, action, reward, next_state, done)
        state = next_state
    agent_annonce.replay(32)