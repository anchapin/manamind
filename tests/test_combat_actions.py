"""Tests for combat-related action validators and executors."""

from manamind.core.action import (
    Action,
    ActionType,
    AssignCombatDamageExecutor,
    AssignCombatDamageValidator,
    DeclareAttackersExecutor,
    DeclareAttackersValidator,
    DeclareBlockersExecutor,
    DeclareBlockersValidator,
    OrderBlockersExecutor,
    OrderBlockersValidator,
)
from manamind.core.game_state import Card, create_empty_game_state


class TestDeclareAttackersValidator:
    """Test DeclareAttackersValidator."""

    def test_valid_attacker_declaration(self):
        """Test valid attacker declaration during combat phase."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create an attacking creature
        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            power=2,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        creature.tapped = False
        creature.summoning_sick = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_not_combat_phase(self):
        """Test attacker declaration outside combat phase."""
        game_state = create_empty_game_state()
        game_state.phase = "main"  # Not combat phase

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_not_active_player(self):
        """Test attacker declaration by non-active player."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 1  # Other player is active
        game_state.priority_player = 0

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_tapped_attacker(self):
        """Test declaring tapped creature as attacker."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create a tapped creature
        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            power=2,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        creature.tapped = True  # Already tapped
        creature.summoning_sick = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_summoning_sick_attacker(self):
        """Test declaring summoning sick creature as attacker."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create a summoning sick creature
        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            power=2,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        creature.tapped = False
        creature.summoning_sick = True  # Has summoning sickness
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_haste_summoning_sick_attacker(self):
        """Test declaring summoning sick creature with haste as attacker."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create a summoning sick creature with haste
        creature = Card(
            name="Lightning Elemental",
            card_type="Creature",
            power=4,
            toughness=1,
            instance_id=1,
            keywords=["Haste"],
        )
        creature.tapped = False
        creature.summoning_sick = True  # Has summoning sickness
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_defender_attacker(self):
        """Test declaring creature with defender as attacker."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create a creature with defender
        creature = Card(
            name="Wall of Stone",
            card_type="Creature",
            power=0,
            toughness=8,
            instance_id=1,
            keywords=["Defender"],
        )
        creature.tapped = False
        creature.summoning_sick = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        validator = DeclareAttackersValidator()
        assert validator.validate(action, game_state) is False


class TestDeclareAttackersExecutor:
    """Test DeclareAttackersExecutor."""

    def test_execute_attacker_declaration(self):
        """Test executing attacker declaration."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create an attacking creature
        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            power=2,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        creature.tapped = False
        creature.summoning_sick = False
        creature.attacking = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1],
        )

        executor = DeclareAttackersExecutor()
        new_state = executor.execute(action, game_state)

        # Find the creature in the new state
        new_creature = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_creature = card
                break

        assert new_creature is not None
        assert new_creature.attacking is True
        assert new_creature.tapped is True  # Attacking creatures become tapped
        assert new_state.priority_player == 1  # Priority passes to defender

    def test_execute_multiple_attackers(self):
        """Test executing multiple attacker declarations."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create multiple attacking creatures
        for i in range(3):
            creature = Card(
                name=f"Creature {i}",
                card_type="Creature",
                power=2,
                toughness=2,
                instance_id=i + 1,
                keywords=[],
            )
            creature.tapped = False
            creature.summoning_sick = False
            creature.attacking = False
            game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.DECLARE_ATTACKERS,
            player_id=0,
            attackers=[1, 2, 3],
        )

        executor = DeclareAttackersExecutor()
        new_state = executor.execute(action, game_state)

        # Check all creatures are attacking and tapped
        attacking_count = 0
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id in [1, 2, 3]:
                assert card.attacking is True
                assert card.tapped is True
                attacking_count += 1

        assert attacking_count == 3


class TestDeclareBlockersValidator:
    """Test DeclareBlockersValidator."""

    def test_valid_blocker_declaration(self):
        """Test valid blocker declaration."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0  # Player 0 is attacking
        game_state.priority_player = 1  # Player 1 has priority (defending)

        # Create attacking creature for player 0
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            power=2,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        # Create blocking creature for player 1
        blocker = Card(
            name="Blocker",
            card_type="Creature",
            power=1,
            toughness=3,
            instance_id=2,
            keywords=[],
        )
        blocker.tapped = False
        game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.DECLARE_BLOCKERS,
            player_id=1,
            blockers={1: [2]},  # Block attacker 1 with blocker 2
        )

        validator = DeclareBlockersValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_not_combat_phase(self):
        """Test blocker declaration outside combat phase."""
        game_state = create_empty_game_state()
        game_state.phase = "main"  # Not combat phase

        action = Action(
            action_type=ActionType.DECLARE_BLOCKERS,
            player_id=1,
            blockers={1: [2]},
        )

        validator = DeclareBlockersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_active_player_blocking(self):
        """Test blocking by the active (attacking) player."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0  # Active player trying to block

        action = Action(
            action_type=ActionType.DECLARE_BLOCKERS,
            player_id=0,
            blockers={1: [2]},
        )

        validator = DeclareBlockersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_tapped_blocker(self):
        """Test declaring tapped creature as blocker."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 1

        # Create attacking creature
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        # Create tapped blocker
        blocker = Card(
            name="Blocker",
            card_type="Creature",
            instance_id=2,
            keywords=[],
        )
        blocker.tapped = True  # Already tapped
        game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.DECLARE_BLOCKERS,
            player_id=1,
            blockers={1: [2]},
        )

        validator = DeclareBlockersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_blocking_multiple_attackers(self):
        """Test using same creature to block multiple attackers."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 1

        # Create multiple attacking creatures
        for i in range(2):
            attacker = Card(
                name=f"Attacker {i}",
                card_type="Creature",
                instance_id=i + 1,
                keywords=[],
            )
            attacker.attacking = True
            game_state.players[0].battlefield.add_card(attacker)

        # Create one blocker
        blocker = Card(
            name="Blocker",
            card_type="Creature",
            instance_id=3,
            keywords=[],
        )
        blocker.tapped = False
        game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.DECLARE_BLOCKERS,
            player_id=1,
            blockers={1: [3], 2: [3]},  # Same blocker blocking both attackers
        )

        validator = DeclareBlockersValidator()
        assert validator.validate(action, game_state) is False


class TestDeclareBlockersExecutor:
    """Test DeclareBlockersExecutor."""

    def test_execute_blocker_declaration(self):
        """Test executing blocker declaration."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 1

        # Create attacking creature
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        attacker.blocked_by = []
        game_state.players[0].battlefield.add_card(attacker)

        # Create blocking creature
        blocker = Card(
            name="Blocker",
            card_type="Creature",
            instance_id=2,
            keywords=[],
        )
        blocker.tapped = False
        blocker.blocking = None
        game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.DECLARE_BLOCKERS,
            player_id=1,
            blockers={1: [2]},
        )

        executor = DeclareBlockersExecutor()
        new_state = executor.execute(action, game_state)

        # Find the blocker in the new state
        new_blocker = None
        for card in new_state.players[1].battlefield.cards:
            if card.instance_id == 2:
                new_blocker = card
                break

        # Find the attacker in the new state
        new_attacker = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_attacker = card
                break

        assert new_blocker is not None
        assert new_blocker.blocking == 1  # Blocking attacker 1
        assert new_attacker is not None
        assert 2 in new_attacker.blocked_by  # Blocked by blocker 2
        assert new_state.priority_player == 0  # Priority back to attacker


class TestAssignCombatDamageValidator:
    """Test AssignCombatDamageValidator."""

    def test_valid_damage_assignment(self):
        """Test valid combat damage assignment."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.priority_player = 0

        # Create attacking creature
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            power=3,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        # Create blocking creature
        blocker = Card(
            name="Blocker",
            card_type="Creature",
            power=2,
            toughness=4,
            instance_id=2,
            keywords=[],
        )
        blocker.blocking = 1
        game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.ASSIGN_COMBAT_DAMAGE,
            player_id=0,
            damage_assignment={
                1: {2: 3}
            },  # Attacker 1 deals 3 damage to blocker 2
        )

        validator = AssignCombatDamageValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_not_combat_phase(self):
        """Test damage assignment outside combat phase."""
        game_state = create_empty_game_state()
        game_state.phase = "main"  # Not combat phase

        action = Action(
            action_type=ActionType.ASSIGN_COMBAT_DAMAGE,
            player_id=0,
            damage_assignment={1: {2: 3}},
        )

        validator = AssignCombatDamageValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_excess_damage(self):
        """Test assigning more damage than creature's power."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.priority_player = 0

        # Create attacking creature with power 2
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            power=2,  # Only has 2 power
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        action = Action(
            action_type=ActionType.ASSIGN_COMBAT_DAMAGE,
            player_id=0,
            damage_assignment={
                1: {2: 5}
            },  # Trying to assign 5 damage with 2 power
        )

        validator = AssignCombatDamageValidator()
        assert validator.validate(action, game_state) is False


class TestAssignCombatDamageExecutor:
    """Test AssignCombatDamageExecutor."""

    def test_execute_damage_assignment(self):
        """Test executing combat damage assignment."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.priority_player = 0

        # Create attacking creature
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            power=3,
            toughness=2,
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        # Create blocking creature
        blocker = Card(
            name="Blocker",
            card_type="Creature",
            power=2,
            toughness=2,  # Will die from 3 damage
            instance_id=2,
            keywords=[],
            counters={"damage": 0},
        )
        blocker.blocking = 1
        game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.ASSIGN_COMBAT_DAMAGE,
            player_id=0,
            damage_assignment={1: {2: 3}},
        )

        executor = AssignCombatDamageExecutor()
        new_state = executor.execute(action, game_state)

        # Blocker should be in graveyard (3 damage >= 2 toughness)
        assert len(new_state.players[1].graveyard.cards) == 1
        assert new_state.players[1].graveyard.cards[0].name == "Blocker"

        # Attacker should no longer be attacking
        new_attacker = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_attacker = card
                break

        assert new_attacker is not None
        assert new_attacker.attacking is False

    def test_execute_damage_to_player(self):
        """Test executing combat damage assignment to player."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.priority_player = 0
        game_state.players[1].life = 20

        # Create attacking creature
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            power=5,
            toughness=2,
            instance_id=10,  # Different ID to avoid conflict
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        action = Action(
            action_type=ActionType.ASSIGN_COMBAT_DAMAGE,
            player_id=0,
            damage_assignment={
                10: {1: 5}
            },  # Attacker 10 deals 5 damage to player 1 (target_id=1)
        )

        executor = AssignCombatDamageExecutor()
        new_state = executor.execute(action, game_state)

        # Player should take damage
        assert new_state.players[1].life == 15  # 20 - 5 = 15


class TestOrderBlockersValidator:
    """Test OrderBlockersValidator."""

    def test_valid_blocker_ordering(self):
        """Test valid blocker ordering."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        # Create attacking creature
        attacker = Card(
            name="Attacker",
            card_type="Creature",
            instance_id=1,
            keywords=[],
        )
        attacker.attacking = True
        game_state.players[0].battlefield.add_card(attacker)

        # Create multiple blocking creatures
        for i in range(2):
            blocker = Card(
                name=f"Blocker {i}",
                card_type="Creature",
                instance_id=i + 2,
                keywords=[],
            )
            blocker.blocking = 1  # Both blocking attacker 1
            game_state.players[1].battlefield.add_card(blocker)

        action = Action(
            action_type=ActionType.ORDER_BLOCKERS,
            player_id=0,
            additional_choices={"1": [2, 3]},  # Order blockers for attacker 1
        )

        validator = OrderBlockersValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_not_combat_phase(self):
        """Test blocker ordering outside combat phase."""
        game_state = create_empty_game_state()
        game_state.phase = "main"  # Not combat phase

        action = Action(
            action_type=ActionType.ORDER_BLOCKERS,
            player_id=0,
            additional_choices={"1": [2, 3]},
        )

        validator = OrderBlockersValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_not_active_player(self):
        """Test blocker ordering by non-active player."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 1  # Other player is active
        game_state.priority_player = 0

        action = Action(
            action_type=ActionType.ORDER_BLOCKERS,
            player_id=0,
            additional_choices={"1": [2, 3]},
        )

        validator = OrderBlockersValidator()
        assert validator.validate(action, game_state) is False


class TestOrderBlockersExecutor:
    """Test OrderBlockersExecutor."""

    def test_execute_blocker_ordering(self):
        """Test executing blocker ordering."""
        game_state = create_empty_game_state()
        game_state.phase = "combat"
        game_state.active_player = 0
        game_state.priority_player = 0

        action = Action(
            action_type=ActionType.ORDER_BLOCKERS,
            player_id=0,
            additional_choices={"1": [2, 3]},  # Order blockers for attacker 1
        )

        executor = OrderBlockersExecutor()
        new_state = executor.execute(action, game_state)

        # Check that order was recorded in history
        assert len(new_state.history) >= 1
        last_action = new_state.history[-1]
        assert last_action["action"] == "order_blockers"
        assert last_action["player"] == 0
        assert last_action["blocker_orders"] == {1: [2, 3]}
