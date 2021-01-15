
# import torch

# from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
# from rlpyt.agents.base import AgentInputs

# from rlpyt.utils.tensor import valid_mean
# from rlpyt.utils.quick_args import save__init__args
# from rlpyt.utils.buffer import buffer_method


# class A2C(PolicyGradientAlgo):
#     """
#     Advantage Actor Critic algorithm (synchronous).  Trains the agent by
#     taking one gradient step on each iteration of samples, with advantages
#     computed by generalized advantage estimation.
#     """

#     def __init__(
#             self,
#             discount=0.99,
#             learning_rate=0.001,
#             value_loss_coeff=0.5,
#             entropy_loss_coeff=0.01,
#             OptimCls=torch.optim.Adam,
#             optim_kwargs=None,
#             clip_grad_norm=1.,
#             initial_optim_state_dict=None,
#             gae_lambda=1,
#             normalize_advantage=False,
#             ):
#         """Saves the input settings."""
#         if optim_kwargs is None:
#             optim_kwargs = dict()
#         save__init__args(locals())

#     def initialize(self, *args, **kwargs):
#         super().initialize(*args, **kwargs)
#         self._batch_size = self.batch_spec.size  # For logging.

#     def optimize_agent(self, itr, samples):
#         """
#         Train the agent on input samples, by one gradient step.
#         """
#         if hasattr(self.agent, "update_obs_rms"):
#             # NOTE: suboptimal--obs sent to device here and in agent(*inputs).
#             self.agent.update_obs_rms(samples.env.observation)
#         self.optimizer.zero_grad()
#         loss, entropy, perplexity = self.loss(samples)
#         loss.backward()
#         grad_norm = torch.nn.utils.clip_grad_norm_(
#             self.agent.parameters(), self.clip_grad_norm)
#         self.optimizer.step()
#         opt_info = OptInfo(
#             loss=loss.item(),
#             gradNorm=torch.tensor(grad_norm).item(),  # backwards compatible,
#             entropy=entropy.item(),
#             perplexity=perplexity.item(),
#         )
#         self.update_counter += 1
#         return opt_info

#     def loss(self, samples):
#         """
#         Computes the training loss: policy_loss + value_loss + entropy_loss.
#         Policy loss: log-likelihood of actions * advantages
#         Value loss: 0.5 * (estimated_value - return) ^ 2
#         Organizes agent inputs from training samples, calls the agent instance
#         to run forward pass on training data, and uses the
#         ``agent.distribution`` to compute likelihoods and entropies.  Valid
#         for feedforward or recurrent agents.
#         """
#         agent_inputs = AgentInputs(
#             observation=samples.env.observation,
#             prev_action=samples.agent.prev_action,
#             prev_reward=samples.env.prev_reward,
#         )
#         if self.agent.recurrent:
#             init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T = 0.
#             # [B,N,H] --> [N,B,H] (for cudnn).
#             init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
#             init_rnn_state = buffer_method(init_rnn_state, "contiguous")
#             dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
#         else:
#             dist_info, value = self.agent(*agent_inputs)
#         # TODO: try to compute everyone on device.
#         return_, advantage, valid = self.process_returns(samples)

#         dist = self.agent.distribution
#         logli = dist.log_likelihood(samples.agent.action, dist_info)
#         pi_loss = - valid_mean(logli * advantage, valid)

#         value_error = 0.5 * (value - return_) ** 2
#         value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

#         entropy = dist.mean_entropy(dist_info, valid)
#         entropy_loss = - self.entropy_loss_coeff * entropy

#         loss = pi_loss + value_loss + entropy_loss

#         perplexity = dist.mean_perplexity(dist_info, valid)

#         return loss, entropy, perplexity


import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs

from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_method

from collections import namedtuple

OptInfoCustom = namedtuple("OptInfoCustom", ["loss", "gradNorm", "base_entropy", "pointer_entropy", "base_perplexity", "pointer_perplexity"])

class A2C(PolicyGradientAlgo):
    """
    Advantage Actor Critic algorithm (synchronous).  Trains the agent by
    taking one gradient step on each iteration of samples, with advantages
    computed by generalized advantage estimation.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            normalize_advantage=False,
            ):
        """Saves the input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size  # For logging.

    def optimize_agent(self, itr, samples):
        """
        Train the agent on input samples, by one gradient step.
        """
        if hasattr(self.agent, "update_obs_rms"):
            # NOTE: suboptimal--obs sent to device here and in agent(*inputs).
            self.agent.update_obs_rms(samples.env.observation)
        self.optimizer.zero_grad()
        loss, base_entropy, pointer_entropy, base_perplexity, pointer_perplexity = self.loss(samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info = OptInfoCustom(
            loss=loss.item(),
            gradNorm=torch.tensor(grad_norm).item(),  # backwards compatible,
            base_entropy=base_entropy.item(),
            pointer_entropy=pointer_entropy.item(),
            base_perplexity=base_perplexity.item(),
            pointer_perplexity=pointer_perplexity.item()
        )
        self.update_counter += 1
        return opt_info

    def loss(self, samples):
        """
        Computes the training loss: policy_loss + value_loss + entropy_loss.
        Policy loss: log-likelihood of actions * advantages
        Value loss: 0.5 * (estimated_value - return) ^ 2
        Organizes agent inputs from training samples, calls the agent instance
        to run forward pass on training data, and uses the
        ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        if self.agent.recurrent:
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T = 0.
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info_base, dist_info_pointer, value = self.agent(*agent_inputs)

        # TODO: try to compute everyone on device.
        return_, advantage, valid = self.process_returns(samples)

        dist_base, dist_pointer = self.agent.categorical_dist, self.agent.gaussian_dist

        action = samples.agent.action
        assert(len(action.shape) == 4)

        B, _, _, D = action.shape
        action = action.view(B, D)
        base_action = action[:,0]
        pointer_action = action[:,-2:]

        logli_base = dist_base.log_likelihood(base_action.unsqueeze(-1).long(), dist_info_base)
        logli_pointer = dist_pointer.log_likelihood(pointer_action, dist_info_pointer)
        pi_loss = - valid_mean((logli_base.squeeze(-1) + logli_pointer) * advantage, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        base_entropy = dist_base.mean_entropy(dist_info_base, valid)
        pointer_entropy = dist_pointer.mean_entropy(dist_info_pointer, valid)
        entropy_loss = - self.entropy_loss_coeff * (base_entropy + pointer_entropy)

        loss = pi_loss + value_loss + entropy_loss

        base_perplexity = dist_base.mean_perplexity(dist_info_base, valid)
        pointer_perplexity = dist_pointer.mean_perplexity(dist_info_pointer, valid)

        return loss, base_entropy, pointer_entropy, base_perplexity, pointer_perplexity