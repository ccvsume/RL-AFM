
import numpy as np
import time
import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sac_torch import Agent
from config import Args
from env import Env



args = Args()
env = Env()
agent = Agent(args)

# run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
# if args.track:
#     import wandb
#
#     wandb.init(
#         project=args.wandb_project_name,
#         entity=args.wandb_entity,
#         sync_tensorboard=True,
#         config=vars(args),
#         name=run_name,
#         save_code=True,
#     )
# writer = SummaryWriter(f"runs/{run_name}")
# writer.add_text(
#     "hyperparameters",
#     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
# )

# Automatic entropy tuning
if args.auto_tune:
    target_entropy = -T.prod(T.Tensor((2,))).item()
    log_alpha = T.zeros(1, requires_grad=True)
    auto_alpha = log_alpha.exp().item()
    a_optimizer = T.optim.Adam([log_alpha], lr=args.beta)
else:
    auto_alpha = args.auto_alpha

def main(last_DeflV,
          DeflV,
          last_ZVolt,
          last_error_sum,
          begin,
          ep_done
          ):
    global env, agent, args, target_entropy, log_alpha, auto_alpha, a_optimizer
    # filename = 'inverted_pendulum.png'
    # figure_file = 'plots/' + filename

    load_checkpoint = False

    if begin == 0:
        env.clear()

    last_state = [last_DeflV, last_ZVolt, env.Kp, env.Ki]

    action, _ = agent.choose_action(last_state)

    im_state, im_reward, error_sum = env.step(action, last_state, DeflV, last_error_sum)     # implement state

    # transition放进RM
    agent.remember(last_state, action, im_reward, im_state, ep_done)

    if not load_checkpoint:

        if agent.memory.mem_cntr < args.batch_size:
            mean_v = 0
            weights = [np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [0,0],
                        [0,0],
                        [0, 0],
                        [0,  0],
                        [0,  0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0,  0],
                        [0, 0]]),np.array([0, 0])]
            loss_a = 0
            loss_c = 0
        else:
            last_state, action, reward, state, done = \
                    agent.memory.sample_buffer(args.batch_size)

            reward = T.tensor(reward, dtype=T.float).to(agent.actor.device)
            done = T.tensor(done).to(agent.actor.device)
            state = T.tensor(state, dtype=T.float).to(agent.actor.device)
            last_state = T.tensor(last_state, dtype=T.float).to(agent.actor.device)
            action = T.tensor(action, dtype=T.float).to(agent.actor.device)

            value = agent.value(last_state).view(-1)
            value_ = agent.target_value(state).view(-1)
            value_[done] = 0.0

            actions, log_probs = agent.actor.sample_normal(last_state, reparameterize=False)
            log_probs = log_probs.view(-1)
            q1_new_policy = agent.critic_1.forward(last_state, actions)
            q2_new_policy = agent.critic_2.forward(last_state, actions)
            # import pdb;
            # pdb.set_trace()
            critic_value = T.min(q1_new_policy, q2_new_policy) - auto_alpha * log_probs.unsqueeze(1)
            critic_value = critic_value.view(-1)

            agent.value.optimizer.zero_grad()
            # pdb.set_trace()
            value_target = critic_value - log_probs
            value_loss = 0.5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            agent.value.optimizer.step()

            actions, log_probs = agent.actor.sample_normal(last_state, reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = agent.critic_1.forward(last_state, actions)
            q2_new_policy = agent.critic_2.forward(last_state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)
            
            actor_loss = auto_alpha * log_probs.unsqueeze(1) - critic_value
            actor_loss = T.mean(actor_loss)
            loss_a = actor_loss.item()
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            if args.auto_tune:
                with T.no_grad():
                    _, log_pi = agent.actor.sample_normal(last_state)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                auto_alpha = log_alpha.exp().item()

            agent.critic_1.optimizer.zero_grad()
            agent.critic_2.optimizer.zero_grad()
            q_hat = agent.scale*reward + agent.gamma*value_
            q1_old_policy = agent.critic_1.forward(last_state, action).view(-1)
            q2_old_policy = agent.critic_2.forward(last_state, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward()
            loss_c = critic_loss.item()
            agent.critic_1.optimizer.step()
            agent.critic_2.optimizer.step()

            agent.update_network_parameters()

    return [im_state[1], error_sum, env.output, im_reward, loss_a, loss_c, env.Kp, env.Ki]

# #
# for i in range(10):
#     print(i)
#     [zv, es, op, rw, al, cl] = main(i+1, i+1, i+1, i+1, i+1, 0)
#     print([zv, es, op, rw, al, cl])


