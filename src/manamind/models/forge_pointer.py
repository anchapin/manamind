"""Real-card network with pointer heads for ForgeEnv decisions (#76).

Cards are encoded from the bridge's JSON (type flags, mana value, color
pips, power/toughness, tapped, sick) plus a hashed name embedding, so new
cards need no vocabulary change. Zones are pooled DeepSets-style. The
policy heads score whatever options Forge offers, so the action space is
variable-length:

* priority: one logit per offered option plus a learned "pass" token
* attack: an independent attack/no-attack logit per eligible creature
* block: per blocker, a pointer over the attackers plus "no block"

Only player-visible information is used (#22).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Decision = Dict[str, Any]

NAME_BUCKETS = 4096
NAME_DIM = 16
TYPE_WORDS = (
    "land",
    "creature",
    "instant",
    "sorcery",
    "enchantment",
    "artifact",
    "planeswalker",
    "basic",
    "legendary",
    "aura",
)
PIPS = ("W", "U", "B", "R", "G", "C", "X")
PHASES = (
    "UNTAP",
    "UPKEEP",
    "DRAW",
    "MAIN1",
    "COMBAT_BEGIN",
    "COMBAT_DECLARE_ATTACKERS",
    "COMBAT_DECLARE_BLOCKERS",
    "COMBAT_FIRST_STRIKE_DAMAGE",
    "COMBAT_DAMAGE",
    "COMBAT_END",
    "MAIN2",
    "END_OF_TURN",
    "CLEANUP",
)
DECISION_TYPES = ("priority", "attack", "block")
CARD_FEATURES = len(TYPE_WORDS) + 5 + len(PIPS) + 1
GLOBAL_FEATURES = 10 + len(PHASES) + 1 + len(DECISION_TYPES)
_COST_SYMBOL = re.compile(r"\{([^}]*)\}")


def name_bucket(name: str) -> int:
    """Stable hash of a card name into ``NAME_BUCKETS`` buckets."""
    return zlib.crc32(name.lower().encode("utf-8")) % NAME_BUCKETS


def card_features(card: Dict[str, Any]) -> List[float]:
    """Numeric features for one card dict from the bridge."""
    type_line = str(card.get("type", "")).lower()
    feats = [1.0 if w in type_line else 0.0 for w in TYPE_WORDS]
    feats += [
        min(float(card.get("cmc", 0)), 15.0) / 10.0,
        float(card.get("power", 0)) / 10.0,
        float(card.get("toughness", 0)) / 10.0,
        1.0 if card.get("tapped") else 0.0,
        1.0 if card.get("sick") else 0.0,
    ]
    pips = dict.fromkeys(PIPS, 0.0)
    generic = 0.0
    for sym in _COST_SYMBOL.findall(str(card.get("cost", ""))):
        if sym.isdigit():
            generic += float(sym)
            continue
        for ch in sym.split("/"):
            if ch in pips:
                pips[ch] += 1.0
    feats += [pips[p] / 5.0 for p in PIPS]
    feats.append(generic / 10.0)
    return feats


def global_features(d: Decision) -> List[float]:
    """Scalar game features from a decision's player-visible view."""
    life = d.get("life", [20, 20])
    lib = d.get("library", [0, 0])
    feats = [
        float(life[0]) / 20.0,
        float(life[1]) / 20.0,
        min(float(d.get("turn", 0)), 40.0) / 20.0,
        1.0 if d.get("active") else 0.0,
        float(len(d.get("hand", []))) / 7.0,
        float(d.get("opp_hand_size", 0)) / 7.0,
        float(lib[0]) / 60.0,
        float(lib[1]) / 60.0,
        float(len(d.get("graveyard", []))) / 20.0,
        float(len(d.get("opp_graveyard", []))) / 20.0,
    ]
    phase = str(d.get("phase", ""))
    feats += [1.0 if phase == p else 0.0 for p in PHASES]
    feats.append(0.0 if phase in PHASES else 1.0)
    feats += [1.0 if d.get("t") == t else 0.0 for t in DECISION_TYPES]
    return feats


@dataclass
class ActOutput:
    """One decision's reply plus what training needs."""

    reply: str
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class ForgePointerNet(nn.Module):
    """Card/state encoder with pointer policy heads and a value head."""

    def __init__(self, card_dim: int = 64, state_dim: int = 128) -> None:
        super().__init__()
        self.card_dim = card_dim
        self.state_dim = state_dim
        self.name_emb = nn.Embedding(NAME_BUCKETS, NAME_DIM)
        self.card_mlp = nn.Sequential(
            nn.Linear(CARD_FEATURES + NAME_DIM, card_dim),
            nn.ReLU(),
            nn.Linear(card_dim, card_dim),
            nn.ReLU(),
        )
        zones = 3 * 2 * card_dim + 2 * NAME_DIM
        self.state_mlp = nn.Sequential(
            nn.Linear(zones + GLOBAL_FEATURES, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.option_proj = nn.Linear(card_dim + 2, card_dim)
        self.pass_token = nn.Parameter(torch.zeros(card_dim))
        self.priority_head = self._scorer(state_dim + card_dim)
        self.attack_head = self._scorer(state_dim + card_dim)
        self.block_head = self._scorer(state_dim + 2 * card_dim)
        self.no_block_head = self._scorer(state_dim + card_dim)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    @staticmethod
    def _scorer(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    # -- encoders ---------------------------------------------------------
    def _device(self) -> torch.device:
        return self.pass_token.device

    def encode_cards(self, cards: Sequence[Dict[str, Any]]) -> torch.Tensor:
        """``[n, card_dim]`` embeddings (``[0, card_dim]`` when empty)."""
        dev = self._device()
        if not cards:
            return torch.zeros(0, self.card_dim, device=dev)
        feats = torch.tensor(
            [card_features(c) for c in cards], dtype=torch.float32, device=dev
        )
        ids = torch.tensor(
            [name_bucket(str(c.get("name", ""))) for c in cards], device=dev
        )
        out: torch.Tensor = self.card_mlp(
            torch.cat([feats, self.name_emb(ids)], dim=-1)
        )
        return out

    def _pool(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.shape[0] == 0:
            return torch.zeros(2 * self.card_dim, device=self._device())
        return torch.cat([emb.mean(0), emb.max(0).values])

    def _names(self, names: Sequence[str]) -> torch.Tensor:
        if not names:
            return torch.zeros(NAME_DIM, device=self._device())
        ids = torch.tensor(
            [name_bucket(str(n)) for n in names], device=self._device()
        )
        mean: torch.Tensor = self.name_emb(ids).mean(0)
        return mean

    def encode_state(self, d: Decision) -> torch.Tensor:
        """``[state_dim]`` embedding of the seat's visible view."""
        parts = [
            self._pool(self.encode_cards(d.get("hand", []))),
            self._pool(self.encode_cards(d.get("battlefield", []))),
            self._pool(self.encode_cards(d.get("opp_battlefield", []))),
            self._names(d.get("graveyard", [])),
            self._names(d.get("opp_graveyard", [])),
            torch.tensor(
                global_features(d), dtype=torch.float32, device=self._device()
            ),
        ]
        state: torch.Tensor = self.state_mlp(torch.cat(parts))
        return state

    # -- heads ------------------------------------------------------------
    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Scalar value in [-1, 1] from the piped seat's view."""
        return torch.tanh(self.value_head(state)).squeeze(-1)

    def priority_logits(
        self, state: torch.Tensor, options: Sequence[Dict[str, Any]]
    ) -> torch.Tensor:
        """``[len(options) + 1]`` logits; the last one is pass."""
        opts: List[torch.Tensor] = [self.pass_token]
        if options:
            cards = self.encode_cards([o.get("card", {}) for o in options])
            flags = torch.tensor(
                [
                    [float(bool(o.get("land"))), float(bool(o.get("spell")))]
                    for o in options
                ],
                device=self._device(),
            )
            emb = self.option_proj(torch.cat([cards, flags], dim=-1))
            opts = list(emb) + opts
        stacked = torch.stack(opts)
        s = state.unsqueeze(0).expand(stacked.shape[0], -1)
        logits: torch.Tensor = self.priority_head(
            torch.cat([s, stacked], -1)
        ).squeeze(-1)
        return logits

    def attack_logits(
        self, state: torch.Tensor, creatures: Sequence[Dict[str, Any]]
    ) -> torch.Tensor:
        """``[n]`` attack logits, one per eligible creature."""
        emb = self.encode_cards(creatures)
        s = state.unsqueeze(0).expand(emb.shape[0], -1)
        logits: torch.Tensor = self.attack_head(
            torch.cat([s, emb], -1)
        ).squeeze(-1)
        return logits

    def block_logits(
        self,
        state: torch.Tensor,
        blockers: Sequence[Dict[str, Any]],
        attackers: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        """``[n_blockers, n_attackers + 1]``; the last column is no block."""
        b = self.encode_cards(blockers)
        a = self.encode_cards(attackers)
        nb, na = b.shape[0], a.shape[0]
        s = state.view(1, 1, -1).expand(nb, na, -1)
        pair = torch.cat(
            [
                s,
                b.unsqueeze(1).expand(nb, na, -1),
                a.unsqueeze(0).expand(nb, na, -1),
            ],
            -1,
        )
        hit = self.block_head(pair).squeeze(-1)
        none = self.no_block_head(
            torch.cat([state.unsqueeze(0).expand(nb, -1), b], -1)
        )
        return torch.cat([hit, none], dim=1)

    # -- acting -----------------------------------------------------------
    def act(self, d: Decision, greedy: bool = False) -> ActOutput:
        """Pick a reply for one ForgeEnv decision."""
        state = self.encode_state(d)
        value = self.value(state)
        t = d.get("t")
        if t == "priority":
            options = d.get("options", [])
            dist = torch.distributions.Categorical(
                logits=self.priority_logits(state, options)
            )
            k = dist.probs.argmax() if greedy else dist.sample()
            idx = int(k)
            reply = "-1" if idx == len(options) else str(idx)
            return ActOutput(reply, dist.log_prob(k), dist.entropy(), value)
        if t == "attack":
            creatures = d.get("options", [])
            if not creatures:
                return ActOutput("", _zero(state), _zero(state), value)
            bern = torch.distributions.Bernoulli(
                logits=self.attack_logits(state, creatures)
            )
            pick = (bern.probs > 0.5).float() if greedy else bern.sample()
            reply = " ".join(str(i) for i, p in enumerate(pick) if p > 0.5)
            return ActOutput(
                reply,
                bern.log_prob(pick).sum(),
                bern.entropy().sum(),
                value,
            )
        if t == "block":
            blockers = d.get("blockers", [])
            attackers = d.get("attackers", [])
            if not blockers or not attackers:
                return ActOutput("", _zero(state), _zero(state), value)
            dist = torch.distributions.Categorical(
                logits=self.block_logits(state, blockers, attackers)
            )
            ch = dist.probs.argmax(-1) if greedy else dist.sample()
            reply = " ".join(
                f"{b}:{int(a)}"
                for b, a in enumerate(ch)
                if int(a) < len(attackers)
            )
            return ActOutput(
                reply,
                dist.log_prob(ch).sum(),
                dist.entropy().sum(),
                value,
            )
        raise ValueError(f"unknown decision type {t!r}")


def _zero(like: torch.Tensor) -> torch.Tensor:
    return like.new_zeros(())


def life_potential(d: Dict[str, Any]) -> float:
    """Life-total lead from the piped seat's view, in units of 20 life."""
    life = d.get("life", [20, 20])
    return (float(life[0]) - float(life[1])) / 20.0


def shaped_returns(
    reward: float,
    potentials: Optional[Sequence[float]] = None,
    gamma: float = 1.0,
    shaping_coef: float = 0.0,
) -> List[float]:
    """Per-decision returns for one game.

    Step ``k`` earns ``shaping_coef * (phi[k+1] - phi[k])`` (life-lead
    change until the next decision) and the last step also earns the game
    result. With ``gamma < 1`` a decision is credited mostly for what
    happens soon after it, instead of every decision in a ~150-decision
    game sharing the same final -1.
    """
    if potentials is None or shaping_coef == 0.0:
        n = len(potentials) if potentials is not None else 0
        rewards = [0.0] * n
    else:
        n = len(potentials)
        rewards = [
            shaping_coef * (potentials[k + 1] - potentials[k])
            for k in range(n - 1)
        ] + [0.0]
    if n == 0:
        return []
    rewards[-1] += float(reward)
    out = [0.0] * n
    g = 0.0
    for k in range(n - 1, -1, -1):
        g = rewards[k] + gamma * g
        out[k] = g
    return out


def actor_critic_loss(
    steps: Sequence[ActOutput],
    reward: float,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    potentials: Optional[Sequence[float]] = None,
    gamma: float = 1.0,
    shaping_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Monte Carlo actor-critic loss for one finished game.

    With the defaults every decision gets the game result as its return
    (no discounting) and the value head is the baseline. ``potentials``
    (one :func:`life_potential` per decision), ``gamma`` and
    ``shaping_coef`` turn on life-lead shaping and discounting (#79).
    Decisions with a single legal reply carry no policy gradient and are
    left out of the policy and entropy terms.
    """
    if not steps:
        raise ValueError("no steps to learn from")
    if potentials is not None and len(potentials) != len(steps):
        raise ValueError("need one potential per step")
    logp = torch.stack([s.log_prob for s in steps])
    ent = torch.stack([s.entropy for s in steps])
    val = torch.stack([s.value for s in steps])
    if potentials is None and gamma == 1.0:
        ret = torch.full_like(val, float(reward))
    else:
        pots = (
            list(potentials) if potentials is not None else [0.0] * len(steps)
        )
        ret = val.new_tensor(shaped_returns(reward, pots, gamma, shaping_coef))
    adv = (ret - val).detach()
    choice = (ent.detach() > 1e-6).float()
    n_choice = choice.sum().clamp(min=1.0)
    policy = -(adv * logp * choice).sum() / n_choice
    value = F.mse_loss(val, ret)
    entropy = (ent * choice).sum() / n_choice
    loss = policy + value_coef * value - entropy_coef * entropy
    stats = {
        "loss": float(loss.detach()),
        "policy": float(policy.detach()),
        "value": float(value.detach()),
        "entropy": float(entropy.detach()),
        "adv_abs": float(adv.abs().mean()),
        "choice_frac": float(choice.mean()),
    }
    return loss, stats


def load_pointer_net(path: str) -> Tuple[ForgePointerNet, Optional[dict]]:
    """Load a checkpoint written by ``train_forge``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ForgePointerNet(**ckpt.get("config", {}))
    net.load_state_dict(ckpt["network"])
    return net, ckpt.get("meta")
