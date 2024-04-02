from config import Args
import torch as T
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from PID import PID

class Agent(PID):
    def __init__(self, arg: Args):
        super().__init__()
        self.gamma = arg.gamma
        self.tau = arg.tau
        self.memory = ReplayBuffer(arg.buffer_size, arg.state_dim, arg.action_dim)
        self.n_actions = arg.action_dim

        self.actor = ActorNetwork(arg.alpha, arg.state_dim, n_actions=arg.action_dim,
                    name='actor', max_action=arg.action_high)
        self.critic_1 = CriticNetwork(arg.beta, arg.state_dim, n_actions=arg.action_dim,
                    name='critic_1')
        self.critic_2 = CriticNetwork(arg.beta, arg.state_dim, n_actions=arg.action_dim,
                    name='critic_2')
        self.value = ValueNetwork(arg.beta, arg.state_dim, name='value')
        self.target_value = ValueNetwork(arg.beta, arg.state_dim, name='target_value')

        self.scale = arg.reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0], log_probs

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    

