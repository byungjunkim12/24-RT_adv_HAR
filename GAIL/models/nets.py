import torch

from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, nHidden, discrete, device) -> None:
        super().__init__()

        self.device = device
        self.net = Sequential(
            Linear(state_dim, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, action_dim, device=self.device),
        ).to(self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim, device=self.device))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1).to(self.device)
            distb = Categorical(probs).to(self.device)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std).to(self.device)
            cov_mtx = torch.eye(self.action_dim).to(self.device) * (std ** 2)
            # print('mean:', torch.numel(mean.view(-1)), torch.isnan(mean.view(-1)).sum().item(), 'covMtx:', cov_mtx.shape, torch.numel(cov_mtx.view(-1)), torch.isnan(cov_mtx.view(-1)).sum().item())

            distb = MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(Module):
    def __init__(self, state_dim, nHidden, device) -> None:
        super().__init__()
        
        self.device = device
        self.net = Sequential(
            Linear(state_dim, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, 1, device=self.device),
        ).to(device)

    def forward(self, states):
        return self.net(states)


class Discriminator(Module):
    def __init__(self, state_dim, action_dim, nHidden, discrete, device) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nHidden = nHidden
        self.discrete = discrete
        self.device = device

        if self.discrete:
            self.act_emb = Embedding(
                action_dim, state_dim
            )
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim

        self.net = Sequential(
            Linear(self.net_in_dim, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, nHidden, device=self.device),
            Tanh(),
            Linear(nHidden, 1, device=self.device),
        ).to(self.device)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        if self.discrete:
            actions = self.act_emb(actions.long())
        

        sa = torch.cat([states, actions], dim=-1)

        return self.net(sa)


class Expert(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action
