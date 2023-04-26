import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Predict_Controllable(nn.Module):

    def __init__(self, state_dim, hidden_dim, latent_dim, n_agent, action_dim, lr=3e-4):
        super(Predict_Controllable, self).__init__()

        self.encode_1 = nn.Linear(state_dim, hidden_dim)
        self.encode_2 = nn.Linear(hidden_dim, hidden_dim)
        self.encode_last_fc = nn.Linear(hidden_dim, latent_dim)

        self.predict_action_1 = nn.Linear(latent_dim * 2 + n_agent, hidden_dim)
        self.predict_action_2 = nn.Linear(hidden_dim + n_agent, hidden_dim)
        self.predict_action_last_fc = nn.Linear(
            hidden_dim + n_agent, action_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.CE = nn.CrossEntropyLoss()

    def encode(self, state):
        h = F.relu(self.encode_1(state))
        h = F.relu(self.encode_2(h))
        return self.encode_last_fc(h)

    def forward(self, state, next_state, agent_id):
        latent = self.encode(state)
        next_latent = self.encode(next_state)
        concate_latent = torch.cat([latent, next_latent, agent_id], dim=-1)

        h = F.relu(self.predict_action_1(concate_latent))
        h = torch.cat([h, agent_id], dim=-1)
        h = F.relu(self.predict_action_2(h))
        h = torch.cat([h, agent_id], dim=-1)
        x = torch.softmax(self.predict_action_last_fc(h), dim=-1)

        return x

    def update(self, state, next_state, agent_id, action):
        predict_action = self.forward(state, next_state, agent_id)
        loss = self.CE(predict_action, action.long())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
