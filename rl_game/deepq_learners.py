import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearner:
    def __init__(self, state_size, action_size, seed, buffer_size=10000, batch_size=64, discount=0.99, lr=0.001, tau=0.001, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount = discount
        self.lr = lr
        self.tau = tau
        self.update_every = update_every

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.sample()
                self.learn(experiences, self.discount)

    def act(self, state, epsilon=0.):
        # epsilon-greedy action selection
        if random.random() > epsilon:
            # send state to pytorch device
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # activate inference mode (e.g. no dropout), disable gradient
            self.qnetwork_local.eval()        
            with torch.no_grad():
                action_q_values = self.qnetwork_local(state)
            # back to training mode
            self.qnetwork_local.train()
            return np.argmax(action_q_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount):
        states, actions, rewards, next_states, dones = experiences
        # get target Q, i.e. Q_targets = r + γ * max(Q(s',a))
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (discount * Q_targets_next * (1 - dones))
        # get local Q, i.e. Q(s,a)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Minimize TD error: backprop for local network to train towards target
        loss = nn.MSELoss()(Q_expected, Q_targets)
        #nn.SmoothL1Loss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        # partial update of target network by adding a fraction of the local networks' weights
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)