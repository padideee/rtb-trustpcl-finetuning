import torch
from torch.distributions import Normal


def fwd_rl(initial_state, prior, gfn, log_reward_fn, exploration_std=None, return_exp = False, beta=1.0):

    states, log_p_posterior, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    log_p_prior = prior.get_trajectory_fwd_off(states.detach(), log_reward_fn).detach()

    
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
    reward = log_r.exp()
    adv = reward - reward.mean()

    kl_loss = (log_p_posterior.sum(-1) - log_p_prior.sum(-1))**2
    kl_div = (log_p_posterior.sum(-1) - log_p_prior.sum(-1)).mean()
    
    reinforce_loss = -adv * log_p_posterior.sum(-1)

    return reinforce_loss, kl_loss, kl_div

def fwd_rl_corrected(initial_state, prior, gfn, log_reward_fn, exploration_std=None, return_exp=False, beta=1.0):
    states, log_p_posterior, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    log_p_prior = prior.get_trajectory_fwd_off(states.detach(), log_reward_fn).detach()

    logq = log_p_posterior.sum(dim=-1)
    logp = log_p_prior.sum(dim=-1)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
        reward = log_r.exp()

    with torch.no_grad():
        w = reward - (1.0 / beta) * (logq - logp)
        w = w - w.mean()

    reinforce_kl_loss = - (w * logq).mean()

    kl_div = (logq - logp).mean()

    return reinforce_kl_loss, kl_div


def fwd_rl_corrected_off_policy(initial_state, prior, gfn, log_reward_fn, exploration_std=None, return_exp=False,
                                beta=1.0, return_behavior_logpf=True):
    states, log_p_posterior, log_pbs, log_fs, logpf_behavior = gfn.get_trajectory_fwd(initial_state, exploration_std,
                                                                                      log_reward_fn,
                                                                                      return_behavior_logpf=return_behavior_logpf)
    log_p_prior = prior.get_trajectory_fwd_off(states.detach(), log_reward_fn).detach()

    logq = log_p_posterior.sum(dim=-1)
    logq_b = logpf_behavior.sum(dim=-1)
    logp = log_p_prior.sum(dim=-1)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
        reward = log_r.exp()

    with torch.no_grad():
        w = reward - (1.0 / beta) * (logq - logp)
        w = w - w.mean()

    is_ratio = torch.softmax((logq - logq_b) / log_p_posterior.shape[1], dim=-1).detach()

    reinforce_kl_loss = - (is_ratio * w * logq).sum()

    kl_div = (is_ratio * (logq - logp)).sum()

    return reinforce_kl_loss, kl_div

def fwd_rtb(initial_state, prior, gfn, log_reward_fn, exploration_std=None, return_exp = False, beta = 1.0):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    log_p_prior = prior.get_trajectory_fwd_off(states.detach(), log_reward_fn).detach()
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
    kl_div = (log_pfs.sum(-1) - log_p_prior.sum(-1)).mean()

    loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_p_prior.sum(-1) - beta * log_r) ** 2)
    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_r, kl_div
    return loss.mean(), kl_div


def bwd_mle(samples, gfn, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(samples, exploration_std, log_reward_fn)
    loss = -log_pfs.sum(-1)
    return loss.mean()


