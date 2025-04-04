import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

class DQNannonce(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNannonce, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --------------------- DQN Agent ---------------------
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=32, memory_size=10000):
        """
        Initialise un agent DQN.
        """
        self.state_size = state_size 
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0           # taux d'exploration initial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.model = DQNannonce(state_size, action_size)
        self.target_model = DQNannonce(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Utilise epsilon-greedy pour choisir une action."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())
    
    def replay(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            target = reward
            if not done:
                with torch.no_grad():
                    target += self.gamma * torch.max(self.target_model(next_state_tensor.unsqueeze(0))).item()
            # Q-value actuel
            state_q = self.model(state_tensor)
            target_f = state_q.clone().detach()
            target_f[action] = target
            states.append(state_tensor)
            targets.append(target_f)
        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
    
    def update_target_model(self):
        """Synchronise le modèle cible avec le modèle principal."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay