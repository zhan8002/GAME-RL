import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.distributions import Categorical, DiagGaussian
from core.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space=None, base=None, base_kwargs=None, use_action_masks=False):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MLPBase


        self.base = base(obs_shape[0], **base_kwargs)
        self.use_action_masks = use_action_masks

        if action_space is not None:
            if action_space.__class__.__name__ == "Discrete":
                num_outputs = action_space.n
                self.dist = Categorical(self.base.output_size, num_outputs)
            elif action_space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
            else:
                raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, action_masks=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        if self.use_action_masks:
            dist = self.dist(actor_features, mask=action_masks)
        else:
            dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, action_masks=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        if self.use_action_masks:
            dist = self.dist(actor_features, mask=action_masks)
        else:
            dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class MultiHeadPolicy(Policy):
    def __init__(self, obs_shape, action_heads, base=None, base_kwargs=None, use_action_masks=False):
        super().__init__(obs_shape=obs_shape, action_space=None, base=base, base_kwargs=base_kwargs,
                         use_action_masks=use_action_masks)
        self.action_heads = action_heads

    def act(self, inputs, rnn_hxs, masks, action_masks, deterministic=False):
        value, action_input_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        actions, action_log_probs, _ = self.action_heads(input=action_input_features, masks=action_masks,
                                                     deterministic=deterministic)
        return value, actions, action_log_probs, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, actions, action_masks):
        value, action_input_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        _, action_log_probs, entropy = self.action_heads(input=action_input_features, masks=action_masks, actions=actions)

        return value, action_log_probs, entropy, rnn_hxs



class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, recurrent_type, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            if recurrent_type == "GRU":
                self.rnn = nn.GRU(recurrent_input_size, hidden_size)
                self._recurrent_type = "GRU"
            elif recurrent_type == "LSTM":
                self.rnn = nn.LSTM(recurrent_input_size, hidden_size)
                self._recurrent_type = "LSTM"
            else:
                raise NotImplementedError
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_type(self):
        return self._recurrent_type

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_rnn(self, x, hxs, masks):
        #masks are for when its done (so have to reset hidden state, basically).
        if self._recurrent_type == "LSTM":
            cxs = hxs[1]
            hxs = hxs[0]
        if x.size(0) == hxs.size(0):
            if self._recurrent_type == "LSTM":
                x, hxs = self.rnn(x.unsqueeze(0), ((hxs * masks).unsqueeze(0), (cxs * masks).unsqueeze(0)))
                cxs = hxs[1].squeeze(0)
                hxs = hxs[0].squeeze(0)
                hxs = (hxs, cxs)
            else:
                x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
                hxs = hxs.squeeze(0)
            x = x.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            #which steps in sequence have a zero
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T] #basically now have list of indices where mask is applied.
            hxs = hxs.unsqueeze(0)
            if self._recurrent_type == "LSTM":
                cxs = cxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros)-1):
                #process steps that don't have any masks together - much faster!!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                if self._recurrent_type == "LSTM":
                    rnn_scores, hxs = self.rnn(
                        x[start_idx:end_idx], ((hxs * masks[start_idx].view(1, -1, 1)), (cxs * masks[start_idx].view(1, -1, 1)))
                    )
                    cxs = hxs[1]
                    hxs = hxs[0]
                else:
                    rnn_scores, hxs = self.rnn(
                        x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                    )
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
            if self._recurrent_type == "LSTM":
                cxs = cxs.squeeze(0)
                hxs = (hxs, cxs)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, recurrent_type="GRU", hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, recurrent_type, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, recurrent_type="GRU", hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, recurrent_type, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # self.rnn_net = nn.LSTM(8, 64, 1, batch_first=True)
        self.rnn_net = nn.GRU(8, 64, 1, batch_first=True)

        self.fc = nn.Linear(64, 64)

        self.fc_p = nn.Linear(64, 312)
        self.fc_v = nn.Linear(64, 1)

        # self.actor = nn.Sequential(
        #     init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        # )
        #
        # self.critic = nn.Sequential(
        #     init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        # )
        #
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        #
        # self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x, rnn_hidden = self.rnn_net(x)

        x = F.relu(self.fc(x))

        hidden = x[:, -1, :]

        # v = self.fc_v(x[-1, :])

        # hidden_critic = self.critic(x)
        # hidden_actor = self.actor(x)

        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)
        #
        # hidden_critic = self.critic(x)
        # hidden_actor = self.actor(x)

        return self.critic_linear(hidden), hidden, rnn_hxs