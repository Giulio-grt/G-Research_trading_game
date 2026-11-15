# Round 3: teams compete for the same reward (exponential, mean 10). If two or
# more teams take the same reward, all of them are locked out and nobody gets it.
#
# The best rewards are the ones every other team goes for, so taking them while
# others are free mostly ends in a collision. Instead, take the second-best
# rewards that the other free teams let go:
#  - Learn what each other team takes from other_scores: a team that was free
#    and then shows up locked took that round's reward.
#  - Value a reward at reward * P(no free team takes it).
#  - Value a lockout with the round 2 rule (renewal rate mid-game, exact
#    end-of-game DP), but on what taking each past reward would have paid us:
#    the reward if no other team took it, 0 otherwise.
# Before a team has been seen, assume it plays the solo-optimal threshold.

import math
import random
from bisect import bisect_left, bisect_right, insort
import numpy as np

MEAN = 10.0     # rewards are exponential with this mean
WARMUP = 10     # resolved rounds to watch before taking anything
ENDGAME = 150   # rounds left at which to switch to the end-of-game rule
# Teams running the same rule decide alike and always collide, so randomise:
JITTER = 0.15   # threshold scaled by a random factor in [1 - JITTER, 1 + JITTER] per game
P_TAKE = 0.6    # chance of taking a reward the rule says to take

rewards = []    # every reward this game; rewards[s - 1] is round s
paid = []       # per resolved round: its reward if no other team took it, else 0
took = []       # took[j]: sorted rewards team j took while free
passed = []     # passed[j]: sorted rewards team j let go while free
until = []      # until[j]: last round team j is known to be locked
prev = None     # other teams' (score, locked) at the previous call
votes = [0, 0]  # evidence the locked flags describe round t (index 0) or round t - 1 (index 1)
last_t = 0
endgame = None  # (lockout, V) where V[n] = expected points with n rounds left
solo = {}       # lockout -> solo-optimal threshold for exponential(MEAN) rewards
scale = 1.0     # this game's random threshold factor


def solo_threshold(lockout):
    if lockout not in solo:
        c = MEAN
        for _ in range(100):
            p = math.exp(-c / MEAN)
            c = lockout * (c + MEAN) * p / (1 + lockout * p)
        solo[lockout] = c
    return solo[lockout]


def take_prob(j, r, prior_c):
    # Share of team j's choices consistent with "takes r": rewards <= r it took
    # against rewards >= r it let go, plus one prior choice at the solo threshold.
    a = bisect_right(took[j], r)
    b = len(passed[j]) - bisect_left(passed[j], r)
    return (a + (r > prior_c)) / (a + b + 1)


def learn(before, now, t, lockout):
    for (s0, _), (s1, l1) in zip(before, now):
        if s1 > s0:  # this team won round t - 1's reward
            votes[0 if l1 else 1] += 1
    lag = votes[1] > votes[0]
    if len(rewards) < 2 + lag:
        return
    s = t - 1 - lag  # the round these lock changes describe
    r = rewards[-2 - lag]
    anyone = False
    for j, ((_, l0), (_, l1)) in enumerate(zip(before, now)):
        if l0:
            continue
        if l1:
            insort(took[j], r)
            until[j] = s + lockout
            anyone = True
        else:
            insort(passed[j], r)
    paid.append(0.0 if anyone else r)


def learned_threshold(lockout):
    x = np.sort(np.asarray(paid))[::-1]
    k = np.arange(1, len(x) + 1)
    g = (np.cumsum(x) / (len(x) + lockout * k)).max()
    return lockout * max(float(g), 0.0)


def endgame_values(lockout):
    xs = sorted(paid)
    m = len(xs)
    above = [0.0] * (m + 1)  # above[i] = sum(xs[i:])
    for i in range(m - 1, -1, -1):
        above[i] = above[i + 1] + xs[i]
    V = [0.0] * (ENDGAME + 1)
    for n in range(1, ENDGAME + 1):
        take = V[n - 1 - lockout] if n - 1 - lockout > 0 else 0.0
        skip = V[n - 1]
        c = skip - take
        # V[n] = E[max(X + take, skip)] = skip + E[max(X - c, 0)]
        i = bisect_right(xs, c)
        V[n] = skip + (above[i] - c * (m - i)) / m
    return V


def play(reward: float, lockout: int, t: int, T: int, your_score: float, other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
    global rewards, paid, took, passed, until, prev, votes, last_t, endgame, scale
    if t <= last_t or last_t == 0:  # a new game started
        rewards, paid, prev, votes, endgame = [], [], None, [0, 0], None
        scale = random.uniform(1 - JITTER, 1 + JITTER)
    last_t = t

    reward = float(reward)
    rewards.append(reward if math.isfinite(reward) else 0.0)
    lockout = max(int(lockout), 0)
    others = [(float(s), bool(l)) for s, l in other_scores]
    if prev is None or len(prev) != len(others):
        took, passed, until = [[] for _ in others], [[] for _ in others], [0] * len(others)
    else:
        learn(prev, others, t, lockout)
    prev = others
    if is_locked or not math.isfinite(reward) or reward <= 0 or len(paid) < WARMUP:
        return False

    n = max(T - t + 1, 1)  # rounds left, including this one
    if n <= ENDGAME:
        if endgame is None or endgame[0] != lockout:
            endgame = (lockout, endgame_values(lockout))
        V = endgame[1]
        threshold = V[n - 1] - (V[n - 1 - lockout] if n - 1 - lockout > 0 else 0.0)
    else:
        threshold = learned_threshold(lockout)

    # Chance that no other team takes this reward too.
    prior_c = solo_threshold(lockout) * min(lockout, n - 1) / max(lockout, 1)
    lag = votes[1] > votes[0]
    free_q = 1.0
    for j, (_, locked) in enumerate(others):
        if not lag:
            free = not locked
        elif locked:  # flag is for round t - 1: free now only if that lock just ended
            free = until[j] == t - 1
        else:  # free in round t - 1: still free unless it took that reward
            free = 1.0 - take_prob(j, rewards[-2], prior_c)
        if free:
            free_q *= 1.0 - free * take_prob(j, reward, prior_c)
    return bool(reward * free_q > scale * threshold and random.random() < P_TAKE)
