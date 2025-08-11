"""Tests for targeting and choice action validators and executors."""

from manamind.core.action import (
    Action,
    ActionType,
    ChooseModeExecutor,
    ChooseModeValidator,
    ChooseTargetExecutor,
    ChooseTargetValidator,
    ChooseXValueExecutor,
    ChooseXValueValidator,
    PayManaExecutor,
    PayManaValidator,
    TapForManaExecutor,
    TapForManaValidator,
)
from manamind.core.game_state import Card, create_empty_game_state


class TestChooseTargetValidator:
    """Test ChooseTargetValidator."""

    def test_valid_target_selection_card(self):
        """Test valid card target selection."""
        game_state = create_empty_game_state()

        # Add target card to battlefield
        target_card = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[1].battlefield.add_card(target_card)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target_card],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_target_selection_player(self):
        """Test valid player target selection."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_players=[1],  # Target opponent
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_multiple_targets(self):
        """Test valid multiple target selection."""
        game_state = create_empty_game_state()

        # Add multiple target cards
        target1 = Card(name="Grizzly Bears", card_type="Creature")
        target2 = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[1].battlefield.add_card(target1)
        game_state.players[1].hand.add_card(target2)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target1, target2],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_no_targets(self):
        """Test target selection with no targets."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[],
            target_players=[],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_target_not_found(self):
        """Test targeting card not in game state."""
        game_state = create_empty_game_state()

        # Card not in any zone
        target_card = Card(name="Grizzly Bears", card_type="Creature")

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target_card],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_target_hexproof(self):
        """Test targeting hexproof creature."""
        game_state = create_empty_game_state()

        # Add hexproof creature
        hexproof_creature = Card(
            name="Hexproof Bear", card_type="Creature", keywords=["Hexproof"]
        )
        game_state.players[1].battlefield.add_card(hexproof_creature)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[hexproof_creature],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_target_shroud(self):
        """Test targeting creature with shroud."""
        game_state = create_empty_game_state()

        # Add creature with shroud
        shroud_creature = Card(
            name="Shroud Bear", card_type="Creature", keywords=["Shroud"]
        )
        game_state.players[1].battlefield.add_card(shroud_creature)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[shroud_creature],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_target_protection(self):
        """Test targeting creature with protection."""
        game_state = create_empty_game_state()

        # Add creature with protection
        protected_creature = Card(
            name="Protected Bear",
            card_type="Creature",
            keywords=["Protection from red"],
        )
        game_state.players[1].battlefield.add_card(protected_creature)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[protected_creature],
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_player_target_out_of_range(self):
        """Test targeting invalid player ID."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_players=[5],  # Invalid player ID
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_targeting_requirements(self):
        """Test valid targeting with requirements."""
        game_state = create_empty_game_state()

        # Add creature target
        target_creature = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[1].battlefield.add_card(target_creature)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target_creature],
            additional_choices={
                "targeting_requirements": {"count": 1, "type": "creature"}
            },
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_targeting_requirements_wrong_type(self):
        """Test invalid targeting with wrong target type."""
        game_state = create_empty_game_state()

        # Add non-creature target
        target_card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[1].hand.add_card(target_card)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target_card],
            additional_choices={
                "targeting_requirements": {
                    "count": 1,
                    "type": "creature",  # Requires creature
                }
            },
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_targeting_requirements_wrong_count(self):
        """Test invalid targeting with wrong target count."""
        game_state = create_empty_game_state()

        # Add creature target
        target_creature = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[1].battlefield.add_card(target_creature)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target_creature],
            additional_choices={
                "targeting_requirements": {
                    "count": 2,  # Requires 2 targets, but only providing 1
                    "type": "creature",
                }
            },
        )

        validator = ChooseTargetValidator()
        assert validator.validate(action, game_state) is False


class TestChooseTargetExecutor:
    """Test ChooseTargetExecutor."""

    def test_execute_target_selection(self):
        """Test executing target selection."""
        game_state = create_empty_game_state()

        # Add target card
        target_card = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[1].battlefield.add_card(target_card)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target_card],
            target_players=[1],
        )

        executor = ChooseTargetExecutor()
        new_state = executor.execute(action, game_state)

        # Check that targets were stored
        assert hasattr(new_state, "pending_targets")
        assert target_card in new_state.pending_targets["cards"]
        assert 1 in new_state.pending_targets["players"]

    def test_execute_multiple_target_selection(self):
        """Test executing multiple target selection."""
        game_state = create_empty_game_state()

        # Add multiple targets
        target1 = Card(name="Grizzly Bears", card_type="Creature")
        target2 = Card(name="Hill Giant", card_type="Creature")
        game_state.players[1].battlefield.add_card(target1)
        game_state.players[1].battlefield.add_card(target2)

        action = Action(
            action_type=ActionType.CHOOSE_TARGET,
            player_id=0,
            target_cards=[target1, target2],
        )

        executor = ChooseTargetExecutor()
        new_state = executor.execute(action, game_state)

        # Check that all targets were stored
        assert len(new_state.pending_targets["cards"]) == 2
        assert target1 in new_state.pending_targets["cards"]
        assert target2 in new_state.pending_targets["cards"]


class TestChooseModeValidator:
    """Test ChooseModeValidator."""

    def test_valid_single_mode_selection(self):
        """Test valid single mode selection."""
        game_state = create_empty_game_state()

        # Create modal spell
        modal_spell = Card(
            name="Charms",
            card_type="Instant",
            oracle_text="Choose one — • Deal 2 damage. • Draw a card.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=["mode_1"],
        )

        validator = ChooseModeValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_multiple_mode_selection(self):
        """Test valid multiple mode selection."""
        game_state = create_empty_game_state()

        # Create modal spell that allows multiple modes
        modal_spell = Card(
            name="Cryptic Command",
            card_type="Instant",
            oracle_text=(
                "Choose two — • Counter spell. • Return permanent. "
                "• Tap creatures. • Draw a card."
            ),
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=["mode_1", "mode_2"],
        )

        validator = ChooseModeValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_no_modes_chosen(self):
        """Test mode selection with no modes chosen."""
        game_state = create_empty_game_state()

        modal_spell = Card(
            name="Charms",
            card_type="Instant",
            oracle_text="Choose one — • Deal 2 damage. • Draw a card.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=[],  # No modes chosen
        )

        validator = ChooseModeValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_no_card(self):
        """Test mode selection without card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=None,  # No card
            modes_chosen=["mode_1"],
        )

        validator = ChooseModeValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_mode_not_available(self):
        """Test selecting unavailable mode."""
        game_state = create_empty_game_state()

        # Create modal spell with only 2 modes
        modal_spell = Card(
            name="Charms",
            card_type="Instant",
            oracle_text="Choose one — • Deal 2 damage. • Draw a card.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=["mode_5"],  # Mode doesn't exist
        )

        validator = ChooseModeValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_too_many_modes_choose_one(self):
        """Test selecting too many modes for 'choose one' spell."""
        game_state = create_empty_game_state()

        modal_spell = Card(
            name="Charms",
            card_type="Instant",
            oracle_text="Choose one — • Deal 2 damage. • Draw a card.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=["mode_1", "mode_2"],  # Too many for 'choose one'
        )

        validator = ChooseModeValidator()
        assert validator.validate(action, game_state) is False


class TestChooseModeExecutor:
    """Test ChooseModeExecutor."""

    def test_execute_mode_selection(self):
        """Test executing mode selection."""
        game_state = create_empty_game_state()

        modal_spell = Card(
            name="Charms",
            card_type="Instant",
            oracle_text="Choose one — • Deal 2 damage. • Draw a card.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=["mode_1"],
        )

        executor = ChooseModeExecutor()
        new_state = executor.execute(action, game_state)

        # Check that modes were stored
        assert hasattr(new_state, "pending_modes")
        assert new_state.pending_modes == ["mode_1"]

    def test_execute_multiple_mode_selection(self):
        """Test executing multiple mode selection."""
        game_state = create_empty_game_state()

        modal_spell = Card(
            name="Cryptic Command",
            card_type="Instant",
            oracle_text=(
                "Choose two — • Counter spell. • Return permanent. "
                "• Tap creatures. • Draw a card."
            ),
        )

        action = Action(
            action_type=ActionType.CHOOSE_MODE,
            player_id=0,
            card=modal_spell,
            modes_chosen=["mode_1", "mode_3"],
        )

        executor = ChooseModeExecutor()
        new_state = executor.execute(action, game_state)

        # Check that all modes were stored
        assert new_state.pending_modes == ["mode_1", "mode_3"]


class TestChooseXValueValidator:
    """Test ChooseXValueValidator."""

    def test_valid_x_value_selection(self):
        """Test valid X value selection."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {"colorless": 5}

        # Create X spell
        x_spell = Card(
            name="Fireball",
            card_type="Sorcery",
            mana_cost="{X}{R}",
            oracle_text="Deal X damage to any target.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=3,
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_x_value_zero(self):
        """Test valid X value of zero."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {"R": 1}

        x_spell = Card(
            name="Fireball",
            card_type="Sorcery",
            mana_cost="{X}{R}",
            oracle_text="Deal X damage to any target.",
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=0,
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_x_value_none(self):
        """Test X value selection with None value."""
        game_state = create_empty_game_state()

        x_spell = Card(
            name="Fireball", card_type="Sorcery", mana_cost="{X}{R}"
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=None,  # No value provided
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_negative_x_value(self):
        """Test negative X value selection."""
        game_state = create_empty_game_state()

        x_spell = Card(
            name="Fireball", card_type="Sorcery", mana_cost="{X}{R}"
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=-1,  # Negative value
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_no_card(self):
        """Test X value selection without card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=None,  # No card
            x_value=3,
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_no_x_in_cost(self):
        """Test X value selection for non-X spell."""
        game_state = create_empty_game_state()

        non_x_spell = Card(
            name="Lightning Bolt",
            card_type="Instant",
            mana_cost="{R}",  # No X in cost
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=non_x_spell,
            x_value=3,
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_insufficient_mana(self):
        """Test X value selection with insufficient mana."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {"R": 1}  # Only 1 red mana

        x_spell = Card(
            name="Fireball", card_type="Sorcery", mana_cost="{X}{R}"
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=5,  # Needs 5 + 1 = 6 mana, but only have 1
        )

        validator = ChooseXValueValidator()
        assert validator.validate(action, game_state) is False


class TestChooseXValueExecutor:
    """Test ChooseXValueExecutor."""

    def test_execute_x_value_selection(self):
        """Test executing X value selection."""
        game_state = create_empty_game_state()

        x_spell = Card(
            name="Fireball", card_type="Sorcery", mana_cost="{X}{R}"
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=3,
        )

        executor = ChooseXValueExecutor()
        new_state = executor.execute(action, game_state)

        # Check that X value was stored
        assert hasattr(new_state, "pending_x_value")
        assert new_state.pending_x_value == 3

    def test_execute_x_value_zero(self):
        """Test executing X value selection with zero."""
        game_state = create_empty_game_state()

        x_spell = Card(
            name="Fireball", card_type="Sorcery", mana_cost="{X}{R}"
        )

        action = Action(
            action_type=ActionType.CHOOSE_X_VALUE,
            player_id=0,
            card=x_spell,
            x_value=0,
        )

        executor = ChooseXValueExecutor()
        new_state = executor.execute(action, game_state)

        assert new_state.pending_x_value == 0


class TestTapForManaValidator:
    """Test TapForManaValidator."""

    def test_valid_tap_basic_land(self):
        """Test valid tapping basic land for mana."""
        game_state = create_empty_game_state()

        land = Card(name="Mountain", card_type="Land", instance_id=1)
        land.tapped = False
        game_state.players[0].battlefield.add_card(land)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=land,
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_tap_mana_creature(self):
        """Test valid tapping mana-producing creature."""
        game_state = create_empty_game_state()

        mana_dork = Card(
            name="Llanowar Elves",
            card_type="Creature",
            oracle_text="{T}: Add {G}.",
            instance_id=1,
        )
        mana_dork.tapped = False
        game_state.players[0].battlefield.add_card(mana_dork)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=mana_dork,
            additional_choices={"mana_ability": "{T}: Add {G}."},
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_no_card(self):
        """Test tapping for mana without card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=None,  # No card
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_card_not_found(self):
        """Test tapping card not on battlefield."""
        game_state = create_empty_game_state()

        land = Card(name="Mountain", card_type="Land", instance_id=1)
        # Card not added to battlefield

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=land,
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_already_tapped(self):
        """Test tapping already tapped permanent."""
        game_state = create_empty_game_state()

        land = Card(name="Mountain", card_type="Land", instance_id=1)
        land.tapped = True  # Already tapped
        game_state.players[0].battlefield.add_card(land)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=land,
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_no_mana_ability(self):
        """Test tapping permanent that can't produce mana."""
        game_state = create_empty_game_state()

        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            oracle_text="",  # No mana ability
            instance_id=1,
        )
        creature.tapped = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=creature,
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_mana_ability_not_on_card(self):
        """Test tapping with specified ability not on card."""
        game_state = create_empty_game_state()

        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            oracle_text="",  # No abilities
            instance_id=1,
        )
        creature.tapped = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=creature,
            additional_choices={
                "mana_ability": "{T}: Add {G}."
            },  # Ability not on card
        )

        validator = TapForManaValidator()
        assert validator.validate(action, game_state) is False


class TestTapForManaExecutor:
    """Test TapForManaExecutor."""

    def test_execute_tap_basic_land(self):
        """Test executing tap basic land for mana."""
        game_state = create_empty_game_state()

        land = Card(name="Mountain", card_type="Land", instance_id=1)
        land.tapped = False
        game_state.players[0].battlefield.add_card(land)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=land,
        )

        executor = TapForManaExecutor()
        new_state = executor.execute(action, game_state)

        # Find land in new state
        new_land = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_land = card
                break

        assert new_land is not None
        assert new_land.tapped is True

        # Check mana pool
        player = new_state.players[0]
        assert hasattr(player, "mana_pool")
        assert player.mana_pool["R"] == 1  # Mountain produces red mana

    def test_execute_tap_different_basic_lands(self):
        """Test executing tap different basic lands."""
        basic_lands = [
            ("Plains", "W"),
            ("Island", "U"),
            ("Swamp", "B"),
            ("Mountain", "R"),
            ("Forest", "G"),
        ]

        for land_name, expected_mana in basic_lands:
            game_state = create_empty_game_state()

            land = Card(name=land_name, card_type="Land", instance_id=1)
            land.tapped = False
            game_state.players[0].battlefield.add_card(land)

            action = Action(
                action_type=ActionType.TAP_FOR_MANA,
                player_id=0,
                card=land,
            )

            executor = TapForManaExecutor()
            new_state = executor.execute(action, game_state)

            player = new_state.players[0]
            assert player.mana_pool[expected_mana] == 1

    def test_execute_tap_mana_creature(self):
        """Test executing tap mana-producing creature."""
        game_state = create_empty_game_state()

        mana_dork = Card(
            name="Llanowar Elves",
            card_type="Creature",
            oracle_text="{T}: Add {G}.",
            instance_id=1,
        )
        mana_dork.tapped = False
        game_state.players[0].battlefield.add_card(mana_dork)

        action = Action(
            action_type=ActionType.TAP_FOR_MANA,
            player_id=0,
            card=mana_dork,
            additional_choices={"mana_ability": "{T}: Add {G}."},
        )

        executor = TapForManaExecutor()
        new_state = executor.execute(action, game_state)

        # Find creature in new state
        new_creature = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_creature = card
                break

        assert new_creature is not None
        assert new_creature.tapped is True

        # Check mana pool
        player = new_state.players[0]
        assert player.mana_pool["G"] == 1


class TestPayManaValidator:
    """Test PayManaValidator."""

    def test_valid_mana_payment(self):
        """Test valid mana payment from pool."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 1,
            "U": 1,
            "B": 0,
            "R": 2,
            "G": 1,
            "colorless": 3,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={"R": 2, "colorless": 1},
        )

        validator = PayManaValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_pay_with_colorless(self):
        """Test paying colored mana with colorless."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 0,
            "U": 0,
            "B": 0,
            "R": 1,
            "G": 0,
            "colorless": 5,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={"R": 2},  # Have 1 R + 5 colorless, can pay 2 R
        )

        validator = PayManaValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_no_mana_payment(self):
        """Test mana payment without specifying payment."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment=None,  # No payment specified
        )

        validator = PayManaValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_insufficient_mana(self):
        """Test mana payment with insufficient mana."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 0,
            "U": 0,
            "B": 0,
            "R": 1,
            "G": 0,
            "colorless": 0,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={"R": 3},  # Need 3 R but only have 1
        )

        validator = PayManaValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_negative_payment(self):
        """Test mana payment with negative values."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 1,
            "U": 1,
            "B": 1,
            "R": 1,
            "G": 1,
            "colorless": 1,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={"R": -1},  # Negative payment
        )

        validator = PayManaValidator()
        assert validator.validate(action, game_state) is False


class TestPayManaExecutor:
    """Test PayManaExecutor."""

    def test_execute_mana_payment(self):
        """Test executing mana payment."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 1,
            "U": 1,
            "B": 1,
            "R": 3,
            "G": 1,
            "colorless": 2,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={"R": 2, "colorless": 1},
        )

        executor = PayManaExecutor()
        new_state = executor.execute(action, game_state)

        # Check remaining mana pool
        player = new_state.players[0]
        assert player.mana_pool["R"] == 1  # 3 - 2 = 1
        assert player.mana_pool["colorless"] == 1  # 2 - 1 = 1
        assert player.mana_pool["W"] == 1  # Unchanged
        assert player.mana_pool["U"] == 1  # Unchanged
        assert player.mana_pool["G"] == 1  # Unchanged

    def test_execute_pay_with_colorless_overflow(self):
        """Test paying with colorless mana when specific color insufficient."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 0,
            "U": 0,
            "B": 0,
            "R": 1,
            "G": 0,
            "colorless": 5,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={"R": 3},  # Need 3 R, have 1 R + 5 colorless
        )

        executor = PayManaExecutor()
        new_state = executor.execute(action, game_state)

        # Should pay 1 from R and 2 from colorless
        player = new_state.players[0]
        assert player.mana_pool["R"] == 0  # Used up
        assert player.mana_pool["colorless"] == 3  # 5 - 2 = 3

    def test_execute_empty_payment(self):
        """Test executing empty mana payment."""
        game_state = create_empty_game_state()
        game_state.players[0].mana_pool = {
            "W": 1,
            "U": 1,
            "B": 1,
            "R": 1,
            "G": 1,
            "colorless": 1,
        }

        action = Action(
            action_type=ActionType.PAY_MANA,
            player_id=0,
            mana_payment={},  # Empty payment
        )

        executor = PayManaExecutor()
        new_state = executor.execute(action, game_state)

        # Mana pool should remain unchanged
        player = new_state.players[0]
        for mana_type in ["W", "U", "B", "R", "G", "colorless"]:
            assert player.mana_pool[mana_type] == 1
