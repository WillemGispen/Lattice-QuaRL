""" Module for losses. """

import torch


def bellman_continuous(Q_fn, Q_target_fn, passive_rates_fn, tau):
    """ Variance of the soft Bellman equation with continuous time and infinite horizon. """

    def loss_fn(state, action, reward, _, new_state, normalization, __):

        batch_size = state.shape[0]
        Q = Q_fn(state).view(batch_size, -1)
        Q = Q.gather(-1, action.unsqueeze(dim=-1)).squeeze()  # select Q of action taken

        Q_target = Q_target_fn(new_state)
        passive_rates = passive_rates_fn(new_state)

        exp_value = torch.sum(
            passive_rates * tau * torch.exp(Q_target), dim=(-1, -2, -3))
        exp_Q0 = exp_value / tau / normalization
        exp_value = exp_value + (1 - tau * passive_rates.sum(dim=(-1, -2, -3))) * exp_Q0
        value = torch.log(exp_value)

        target = reward + value

        loss = (Q - target).var()

        return loss

    return loss_fn


def bellman_terminal(Q_fn, Q_target_fn, passive_rates_fn, _):
    """ Mean square error of the soft Bellman equation with discrete time and terminal states. """

    def loss_fn(state, action, reward, done, new_state, normalization, _):

        batch_size = state.shape[0]
        Q = Q_fn(state).view(batch_size, -1)
        Q = Q.gather(-1, action.unsqueeze(dim=-1)).squeeze()  # select Q of action taken

        Q_target = Q_target_fn(new_state)

        passive_rates = passive_rates_fn(new_state)

        exp_value = torch.sum(
            passive_rates * torch.exp(Q_target), dim=(-1, -2, -3))
        exp_value /= normalization
        value = torch.log(exp_value)
        value = value * (1 - done)

        target = reward + value

        # if torch.isnan(reward).any():
        #     raise Exception("nan")

        if torch.isnan(value).any():
            raise Exception("nan")

        loss = ((Q - target)**2).mean()

        return loss

    return loss_fn


def bellman_discrete(Q_fn, Q_target_fn, passive_rates_fn, C):
    """ Variance of the soft Bellman equation with discrete time and infinite horizon. """

    def loss_fn(state, action, reward, _, new_state, normalization, E):

        batch_size = state.shape[0]
        Q = Q_fn(state).view(batch_size, -1)
        Q = Q.gather(-1, action.unsqueeze(dim=-1)).squeeze()  # select Q of action taken

        Q_target = Q_target_fn(new_state)
        passive_rates = passive_rates_fn(new_state)

        exp_value = torch.sum(
            passive_rates * torch.exp(Q_target), dim=(-1, -2, -3))
        Hss = C - normalization + passive_rates.sum(dim=(-1, -2, -3))
        L = state.shape[-1]
        exp_Q0 = exp_value / (Hss - E * L**2)
        exp_value = exp_value + (C - Hss) * exp_Q0
        exp_value /= normalization
        value = torch.log(exp_value)

        target = reward + value

        loss = (Q - target).var()

        return loss

    return loss_fn
