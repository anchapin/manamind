"""A deliberately small, fully-specified subset of Magic.

Full Magic hides a broken training loop indefinitely: when an agent fails to
improve there are a dozen innocent explanations (lossy action encoding, weak
card-text embeddings, rules gaps, the game is simply hard), so a silent bug
survives. On a game small enough to reason about completely, "the agent did
not improve" has exactly one meaning.

The subset, per issue #20:

* Basic lands and vanilla creatures only. No instants, sorceries, abilities.
* No stack. Everything resolves immediately.
* Two identical decks.
* Combat is declare attackers, declare blockers, damage. No tricks, so no
  priority windows.
* 20 life, mulligans off.

It still contains real decisions: when to attack, what to block, how to
trade, when to hold creatures back. This reuses the project's own
``GameState``, ``Card``, ``Action`` and ``ActionType``, so the self-play
machinery under test is the real one, not a parallel toy.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from manamind.core.action import Action, ActionType
from manamind.core.game_state import Card, GameState, Player

if TYPE_CHECKING:  # pragma: no cover
    from manamind.models.policy_value_network import PolicyValueNetwork

# Phases the simple mode cycles through, in order. There is no upkeep or
# end-step interaction to model, so those phases are skipped entirely.
SIMPLE_PHASES = ("main", "combat", "end")

STARTING_HAND_SIZE = 7
STARTING_LIFE = 20


def _land(name: str = "Forest") -> Card:
    return Card(name=name, card_types=["Land"], supertypes=["Basic"])


def _creature(name: str, cost: int, power: int, toughness: int) -> Card:
    return Card(
        name=name,
        mana_cost=f"{{{cost}}}",
        converted_mana_cost=cost,
        card_types=["Creature"],
        power=power,
        toughness=toughness,
        base_power=power,
        base_toughness=toughness,
    )


#: The entire legal card pool. Vanilla creatures across a cheap-to-expensive
#: curve, so mana development is a real decision, plus one basic land.
SIMPLE_CARD_POOL: Tuple[Card, ...] = (
    _creature("Grizzly Bears", 2, 2, 2),
    _creature("Hill Giant", 4, 3, 3),
    _creature("Craw Wurm", 6, 6, 4),
    _creature("Runeclaw Bear", 2, 2, 2),
    _creature("Canopy Spider", 3, 1, 3),
    _land("Forest"),
)


def build_simple_deck() -> List[Card]:
    """Build the fixed 40-card deck both players use.

    Seventeen lands and twenty-three creatures is roughly limited-format
    ratio, which keeps games from stalling on mana in either direction.
    """
    deck: List[Card] = []
    for _ in range(17):
        deck.append(_land())
    curve = [
        ("Grizzly Bears", 7),
        ("Runeclaw Bear", 6),
        ("Canopy Spider", 5),
        ("Hill Giant", 3),
        ("Craw Wurm", 2),
    ]
    by_name = {c.name: c for c in SIMPLE_CARD_POOL}
    for name, count in curve:
        for _ in range(count):
            deck.append(by_name[name].model_copy(deep=True))
    return deck


def create_simple_game_start(seed: Optional[int] = None) -> GameState:
    """Deal a fresh simple-mode game.

    Both libraries are shuffled with the supplied seed and both players draw
    an opening hand, so the same seed always produces the same game. Unlike
    ``create_standard_game_start`` this does not hand back an empty state.
    """
    rng = random.Random(seed)
    players = []
    for player_id in (0, 1):
        player = Player(player_id=player_id, life=STARTING_LIFE)
        deck = build_simple_deck()
        rng.shuffle(deck)
        for card in deck:
            player.library.add_card(card)
        for _ in range(STARTING_HAND_SIZE):
            drawn = player.library.cards.pop(0)
            player.hand.add_card(drawn)
        players.append(player)

    return GameState(
        players=(players[0], players[1]),
        turn_number=1,
        phase="main",
        active_player=0,
        priority_player=0,
        game_mode="simple",
    )


class SimpleRules:
    """The rules engine for the subset.

    Deliberately stateless: every method takes a ``GameState`` and either
    reads it or returns a new one, matching how ``Action.execute`` behaves in
    the main engine.
    """

    # ----------------------------------------------------------------- mana

    @staticmethod
    def available_mana(player: Player) -> int:
        """Untapped lands on the battlefield. One land taps for one mana."""
        return sum(
            1
            for card in player.battlefield.cards
            if card.is_land() and not card.tapped
        )

    @staticmethod
    def _castable(player: Player, card: Card) -> bool:
        return (
            card.is_creature()
            and card.converted_mana_cost <= SimpleRules.available_mana(player)
        )

    # ---------------------------------------------------------------- legal

    @staticmethod
    def legal_actions(game_state: GameState) -> List[Action]:
        """Every legal action for the player who currently has priority.

        Only the active player holds priority in this subset except during
        declare-blockers, which keeps the action list small and the turn
        structure easy to verify.
        """
        actions: List[Action] = []
        pid = game_state.priority_player
        player = game_state.players[pid]

        if game_state.phase == "main":
            if player.can_play_land():
                for card in player.hand.cards:
                    if card.is_land():
                        actions.append(
                            Action(
                                action_type=ActionType.PLAY_LAND,
                                player_id=pid,
                                card=card,
                            )
                        )
                        break  # lands are fungible; one choice is enough
            seen = set()
            for card in player.hand.cards:
                if not SimpleRules._castable(player, card):
                    continue
                key = (card.name, card.converted_mana_cost)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(
                    Action(
                        action_type=ActionType.CAST_SPELL,
                        player_id=pid,
                        card=card,
                    )
                )

        elif game_state.phase == "combat":
            if pid == game_state.active_player:
                candidates = SimpleRules._possible_attackers(player)
                for subset in SimpleRules._attack_options(candidates):
                    actions.append(
                        Action(
                            action_type=ActionType.DECLARE_ATTACKERS,
                            player_id=pid,
                            attackers=list(subset),
                        )
                    )
            else:
                attackers = SimpleRules._attacking_indices(
                    game_state.players[game_state.active_player]
                )
                for assignment in SimpleRules._block_options(
                    player, attackers
                ):
                    actions.append(
                        Action(
                            action_type=ActionType.DECLARE_BLOCKERS,
                            player_id=pid,
                            blockers=assignment,
                        )
                    )

        actions.append(
            Action(action_type=ActionType.PASS_PRIORITY, player_id=pid)
        )
        return actions

    @staticmethod
    def _possible_attackers(player: Player) -> List[int]:
        return [
            idx
            for idx, card in enumerate(player.battlefield.cards)
            if card.is_creature()
            and not card.tapped
            and not card.summoning_sick
        ]

    @staticmethod
    def _attacking_indices(player: Player) -> List[int]:
        return [
            idx
            for idx, card in enumerate(player.battlefield.cards)
            if card.attacking
        ]

    @staticmethod
    def _attack_options(candidates: List[int]) -> List[List[int]]:
        """Attack with everything, or with each creature alone.

        Enumerating every subset explodes; these are the choices that carry
        the strategy in a vanilla board.
        """
        if not candidates:
            return []
        options: List[List[int]] = [list(candidates)]
        if len(candidates) > 1:
            for idx in candidates:
                options.append([idx])
        return options

    @staticmethod
    def _block_options(
        defender: Player, attackers: List[int]
    ) -> List[Dict[int, List[int]]]:
        """Candidate block assignments.

        The full assignment space is exponential. The engine offers no
        blocks, each single blocker against each attacker, and the
        one-for-one line when the boards are the same size, which covers
        chump, trade, or take it.
        """
        if not attackers:
            return []
        available = [
            idx
            for idx, card in enumerate(defender.battlefield.cards)
            if card.is_creature() and not card.tapped
        ]
        options: List[Dict[int, List[int]]] = [{}]
        for attacker in attackers:
            for blocker in available:
                options.append({attacker: [blocker]})
        if len(attackers) == len(available) and len(available) > 1:
            options.append({a: [b] for a, b in zip(attackers, available)})
        return options

    # ---------------------------------------------------------------- apply

    @staticmethod
    def apply(game_state: GameState, action: Action) -> GameState:
        """Apply an action and advance the game as far as it should go."""
        state = game_state.copy()
        player = state.players[action.player_id]

        if action.action_type == ActionType.PLAY_LAND:
            card = SimpleRules._find_in_hand(player, action.card)
            if card is not None:
                player.hand.remove_card(card)
                player.battlefield.add_card(card)
                player.lands_played_this_turn += 1

        elif action.action_type == ActionType.CAST_SPELL:
            card = SimpleRules._find_in_hand(player, action.card)
            if card is not None and SimpleRules._castable(player, card):
                SimpleRules._tap_for(player, card.converted_mana_cost)
                player.hand.remove_card(card)
                card.summoning_sick = True
                player.battlefield.add_card(card)

        elif action.action_type == ActionType.DECLARE_ATTACKERS:
            for idx in action.attackers:
                if 0 <= idx < player.battlefield.size():
                    creature = player.battlefield.cards[idx]
                    creature.attacking = True
                    creature.tapped = True
            state.priority_player = 1 - state.active_player
            return state

        elif action.action_type == ActionType.DECLARE_BLOCKERS:
            state = SimpleRules._resolve_combat(state, action.blockers)
            return SimpleRules._advance_phase(state)

        elif action.action_type == ActionType.PASS_PRIORITY:
            if (
                state.phase == "combat"
                and state.priority_player != state.active_player
            ):
                state = SimpleRules._resolve_combat(state, {})
            return SimpleRules._advance_phase(state)

        return state

    @staticmethod
    def _find_in_hand(player: Player, card: Optional[Card]) -> Optional[Card]:
        if card is None:
            return None
        for candidate in player.hand.cards:
            if (
                candidate.name == card.name
                and candidate.converted_mana_cost == card.converted_mana_cost
            ):
                return candidate
        return None

    @staticmethod
    def _tap_for(player: Player, amount: int) -> None:
        remaining = amount
        for card in player.battlefield.cards:
            if remaining <= 0:
                break
            if card.is_land() and not card.tapped:
                card.tapped = True
                remaining -= 1

    @staticmethod
    def _resolve_combat(
        state: GameState, blocks: Dict[int, List[int]]
    ) -> GameState:
        attacker_player = state.players[state.active_player]
        defender_player = state.players[1 - state.active_player]

        dead_attackers: List[Card] = []
        dead_blockers: List[Card] = []

        for idx, creature in enumerate(attacker_player.battlefield.cards):
            if not creature.attacking:
                continue
            blocker_indices = blocks.get(idx, [])
            blockers = [
                defender_player.battlefield.cards[b]
                for b in blocker_indices
                if 0 <= b < defender_player.battlefield.size()
            ]
            if not blockers:
                defender_player.life -= creature.current_power() or 0
                continue
            damage_left = creature.current_power() or 0
            for blocker in blockers:
                toughness = blocker.current_toughness() or 0
                if damage_left >= toughness:
                    dead_blockers.append(blocker)
                damage_left -= toughness
                if (blocker.current_power() or 0) >= (
                    creature.current_toughness() or 0
                ):
                    dead_attackers.append(creature)

        for card in dead_attackers:
            if attacker_player.battlefield.remove_card(card):
                attacker_player.graveyard.add_card(card)
        for card in dead_blockers:
            if defender_player.battlefield.remove_card(card):
                defender_player.graveyard.add_card(card)

        for creature in attacker_player.battlefield.cards:
            creature.attacking = False

        return state

    # --------------------------------------------------------------- phases

    @staticmethod
    def _advance_phase(state: GameState) -> GameState:
        if state.is_game_over():
            return state

        idx = SIMPLE_PHASES.index(state.phase)
        if idx + 1 < len(SIMPLE_PHASES):
            state.phase = SIMPLE_PHASES[idx + 1]
            state.priority_player = state.active_player
            return state

        return SimpleRules._begin_turn(state, 1 - state.active_player)

    @staticmethod
    def _begin_turn(state: GameState, player_id: int) -> GameState:
        state.active_player = player_id
        state.priority_player = player_id
        state.phase = "main"
        state.turn_number += 1

        player = state.players[player_id]
        player.lands_played_this_turn = 0
        for card in player.battlefield.cards:
            card.tapped = False
            card.summoning_sick = False
            card.attacking = False

        # Draw for turn. Decking is a loss, which is the only way a stalled
        # board eventually resolves.
        if player.library.size() == 0:
            player.life = 0
        else:
            player.hand.add_card(player.library.cards.pop(0))

        return state


class SimpleStateEncoder(nn.Module):
    """A compact encoder for the subset.

    The production ``GameStateEncoder`` pads all six zones to 200 cards and
    runs an LSTM over each, which costs well over a second per state on CPU
    and makes any search-based experiment impossible. The subset has no card
    text, no abilities and no stack, so everything that matters fits in a
    handful of numbers per player.

    Features per player: life, hand size, library size, graveyard size,
    lands in play, untapped lands, creature count, total power, total
    toughness, untapped creature count. Plus a phase one-hot and flags for
    who is active and who holds priority, all from the perspective of the
    player to act.
    """

    PER_PLAYER = 10
    FEATURE_DIM = 2 * PER_PLAYER + len(SIMPLE_PHASES) + 2

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def _player_features(player: Player) -> List[float]:
        creatures = [c for c in player.battlefield.cards if c.is_creature()]
        lands = [c for c in player.battlefield.cards if c.is_land()]
        return [
            player.life / 20.0,
            player.hand.size() / 7.0,
            player.library.size() / 40.0,
            player.graveyard.size() / 10.0,
            len(lands) / 10.0,
            sum(1 for c in lands if not c.tapped) / 10.0,
            len(creatures) / 10.0,
            sum(c.current_power() or 0 for c in creatures) / 20.0,
            sum(c.current_toughness() or 0 for c in creatures) / 20.0,
            sum(1 for c in creatures if not c.tapped) / 10.0,
        ]

    def features(self, game_state: GameState) -> torch.Tensor:
        mover = game_state.priority_player
        values: List[float] = []
        values.extend(self._player_features(game_state.players[mover]))
        values.extend(self._player_features(game_state.players[1 - mover]))
        phase = [
            1.0 if game_state.phase == name else 0.0 for name in SIMPLE_PHASES
        ]
        values.extend(phase)
        values.append(1.0 if game_state.active_player == mover else 0.0)
        values.append(min(game_state.turn_number, 60) / 60.0)
        return torch.tensor(values, dtype=torch.float32)

    def forward(self, game_state: GameState) -> torch.Tensor:
        return self.projection(self.features(game_state))


def build_simple_network(
    action_space_size: int,
    state_dim: int = 32,
    hidden_dim: int = 64,
) -> "PolicyValueNetwork":
    """A right-sized network for the subset: small and fast to search."""
    from manamind.models.policy_value_network import PolicyValueNetwork

    return PolicyValueNetwork(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_residual_blocks=2,
        num_attention_heads=2,
        action_space_size=action_space_size,
        use_attention=False,
        state_encoder=SimpleStateEncoder(output_dim=state_dim),
    )
