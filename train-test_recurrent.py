import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from Modele_ppo.modele_jeu_v2 import CoincheEnv


def train_agent():
    """Entraîne l'IA à jouer à la Coinche avec un LSTM."""
    env = CoincheEnv()

    model = RecurrentPPO(
        "MlpLstmPolicy",  # ✅ Utiliser la politique LSTM
        env,
        verbose=1,
        tensorboard_log="./ppo_coinche_logs",
        gamma=0.99,               
        n_steps=512,              # ⚠️ Doit être un multiple de batch_size
        batch_size=128,           # ⚠️ Doit être ≤ n_steps
        learning_rate=3e-4,       
        ent_coef=0.01,            
        clip_range=0.1,           
        gae_lambda=0.95,          
        vf_coef=0.5,              
        max_grad_norm=0.5,        
        # target_kl=0.1,
        policy_kwargs={
            "net_arch": [256, 256],  # Réseau de neurones pour les couches denses
            "lstm_hidden_size": 128,  # Taille du LSTM
            "n_lstm_layers": 1,  # Nombre de couches LSTM
            "shared_lstm": False  # Si True, le même LSTM est utilisé pour acteur/critic
        },
    )

    print("Démarrage de l'entraînement...")
    model.learn(total_timesteps=1000, tb_log_name="PPO_1", log_interval=None)  # Ajuster selon la puissance de calcul
    model.save("coinche_ia_v2")
    # for i in range(10):
    #     model.learn(total_timesteps=500_000)
    #     model.save(f"coinche_ia_v2_{i}")

    print("Entraînement terminé !")

def test_agent():
    """Teste l'IA après entraînement."""
    env = CoincheEnv()
    model = PPO.load("coinche_ia_v2")

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    # print(f"Récompense moyenne : {mean_reward} ± {std_reward}")
    obs = env.reset()
    done = False
    for player in env.game.players:
        print([str(card) for card in player.get_card()])
    print(env.game.table.get_current_bid().get_trump_suit())
    print(env.game.table.get_current_bid().get_player())
    
    err = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action,training=False)
        if reward == -50:
            err += 1
        env.render()
    print(f"Erreur : {err}")

def choose_valid_action(model, obs, legal_actions):
    action, _ = model.predict(obs)  # Prédiction brute

    # Vérifier si l'action est valide
    if action not in legal_actions:
        action = np.random.choice(legal_actions)  
    
    return action


if __name__ == "__main__":
    train_agent()
    test_agent()
