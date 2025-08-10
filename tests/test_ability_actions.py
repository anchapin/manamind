"""Tests for ability activation validators and executors."""

from manamind.core.action import (
    Action,
    ActionType,
    ActivateAbilityExecutor,
    ActivateAbilityValidator,
    ActivateManaAbilityExecutor,
    ActivateManaAbilityValidator,
)
from manamind.core.game_state import Card, create_empty_game_state


class TestActivateAbilityValidator:
    """Test ActivateAbilityValidator."""

    def test_valid_ability_activation(self):
        """Test valid activated ability activation."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        # Create a creature with an activated ability
        creature = Card(
            name="Prodigal Pyromancer",
            card_type="Creature",
            oracle_text="{T}: Deal 1 damage to any target.",
            instance_id=1,
            keywords=[],
        )
        creature.tapped = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_no_priority(self):
        """Test ability activation without priority."""
        game_state = create_empty_game_state()
        game_state.priority_player = 1  # Other player has priority

        creature = Card(
            name="Prodigal Pyromancer",
            card_type="Creature",
            oracle_text="{T}: Deal 1 damage to any target.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_no_source_card(self):
        """Test ability activation without source card."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=None,  # No source card
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_card_not_found(self):
        """Test ability activation with card not in game state."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        # Create a creature but don't add it to the game state
        creature = Card(
            name="Prodigal Pyromancer",
            card_type="Creature",
            oracle_text="{T}: Deal 1 damage to any target.",
            instance_id=1,
            keywords=[],
        )

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_ability_not_on_card(self):
        """Test ability activation with ability not on card."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        creature = Card(
            name="Grizzly Bears",
            card_type="Creature",
            oracle_text="",  # No abilities
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_graveyard_ability(self):
        """Test activating ability from graveyard (like flashback)."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        # Create a card with flashback in graveyard
        spell = Card(
            name="Flashback Spell",
            card_type="Sorcery",
            oracle_text=(
                "Deal 3 damage. Flashback {2}{R}: You may cast this "
                "card from your graveyard for its flashback cost."
            ),
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].graveyard.add_card(spell)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=spell,
            additional_choices={
                "ability_text": (
                    "Flashback {2}{R}: You may cast this card from "
                    "your graveyard for its flashback cost."
                )
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_tapped_creature_tap_ability(self):
        """Test activating tap ability on already tapped creature."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        creature = Card(
            name="Prodigal Pyromancer",
            card_type="Creature",
            oracle_text="{T}: Deal 1 damage to any target.",
            instance_id=1,
            keywords=[],
        )
        creature.tapped = True  # Already tapped
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_sacrifice_ability(self):
        """Test activating sacrifice ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        creature = Card(
            name="Sakura-Tribe Elder",
            card_type="Creature",
            oracle_text="Sacrifice Sakura-Tribe Elder: Search land.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "Sacrifice Sakura-Tribe Elder: Search land."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_insufficient_mana(self):
        """Test activating ability without sufficient mana."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0
        game_state.players[0].mana_pool = {}  # No mana

        creature = Card(
            name="Expensive Creature",
            card_type="Creature",
            oracle_text="{5}: Draw a card.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={"ability_text": "{5}: Draw a card."},
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_discard_ability(self):
        """Test activating ability with discard cost."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        # Add cards to hand for discard cost
        for i in range(2):
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        creature = Card(
            name="Wild Mongrel",
            card_type="Creature",
            oracle_text="Discard a card: Wild Mongrel gets +1/+1.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "Discard a card: Wild Mongrel gets +1/+1."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_discard_no_cards(self):
        """Test activating discard ability with no cards in hand."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0
        # No cards in hand

        creature = Card(
            name="Wild Mongrel",
            card_type="Creature",
            oracle_text="Discard a card: Wild Mongrel gets +1/+1.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "Discard a card: Wild Mongrel gets +1/+1."
            },
        )

        validator = ActivateAbilityValidator()
        assert validator.validate(action, game_state) is False


class TestActivateAbilityExecutor:
    """Test ActivateAbilityExecutor."""

    def test_execute_tap_ability(self):
        """Test executing tap ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        creature = Card(
            name="Prodigal Pyromancer",
            card_type="Creature",
            oracle_text="{T}: Deal 1 damage to any target.",
            instance_id=1,
            keywords=[],
        )
        creature.tapped = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        executor = ActivateAbilityExecutor()
        new_state = executor.execute(action, game_state)

        # Find creature in new state
        new_creature = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_creature = card
                break

        assert new_creature is not None
        assert new_creature.tapped is True  # Should be tapped after activation
        assert len(new_state.stack) == 1  # Ability should be on stack
        assert new_state.stack[0]["type"] == "activated_ability"

    def test_execute_sacrifice_ability(self):
        """Test executing sacrifice ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        creature = Card(
            name="Sakura-Tribe Elder",
            card_type="Creature",
            oracle_text="Sacrifice Sakura-Tribe Elder: Search land.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "Sacrifice Sakura-Tribe Elder: Search land."
            },
        )

        executor = ActivateAbilityExecutor()
        new_state = executor.execute(action, game_state)

        # Creature should be in graveyard
        assert len(new_state.players[0].battlefield.cards) == 0
        assert len(new_state.players[0].graveyard.cards) == 1
        assert (
            new_state.players[0].graveyard.cards[0].name
            == "Sakura-Tribe Elder"
        )

    def test_execute_discard_ability(self):
        """Test executing discard ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        # Add cards to hand
        for i in range(3):
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        creature = Card(
            name="Wild Mongrel",
            card_type="Creature",
            oracle_text="Discard a card: Wild Mongrel gets +1/+1.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "Discard a card: Wild Mongrel gets +1/+1."
            },
        )

        executor = ActivateAbilityExecutor()
        new_state = executor.execute(action, game_state)

        # Should have one less card in hand and one more in graveyard
        assert len(new_state.players[0].hand.cards) == 2  # 3 - 1 = 2
        assert len(new_state.players[0].graveyard.cards) == 1

    def test_execute_mana_cost_ability(self):
        """Test executing ability with mana cost."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        # Add mana-producing lands
        for i in range(3):
            land = Card(
                name="Mountain",
                card_type="Land",
                instance_id=i + 10,
            )
            land.tapped = False
            game_state.players[0].battlefield.add_card(land)

        creature = Card(
            name="Expensive Creature",
            card_type="Creature",
            oracle_text="{2}: Draw a card.",
            instance_id=1,
            keywords=[],
        )
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={"ability_text": "{2}: Draw a card."},
        )

        executor = ActivateAbilityExecutor()
        new_state = executor.execute(action, game_state)

        # Should have tapped 2 lands for mana
        tapped_lands = 0
        for card in new_state.players[0].battlefield.cards:
            if card.card_type == "Land" and card.tapped:
                tapped_lands += 1

        assert tapped_lands == 2


class TestActivateManaAbilityValidator:
    """Test ActivateManaAbilityValidator."""

    def test_valid_basic_land_mana_ability(self):
        """Test valid basic land mana ability activation."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        land = Card(
            name="Mountain",
            card_type="Land",
            instance_id=1,
            keywords=[],
        )
        land.tapped = False
        game_state.players[0].battlefield.add_card(land)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=land,
        )

        validator = ActivateManaAbilityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_tapped_land(self):
        """Test mana ability activation on tapped land."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        land = Card(
            name="Mountain",
            card_type="Land",
            instance_id=1,
            keywords=[],
        )
        land.tapped = True  # Already tapped
        game_state.players[0].battlefield.add_card(land)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=land,
        )

        validator = ActivateManaAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_explicit_mana_ability(self):
        """Test explicit mana ability activation."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        mana_dork = Card(
            name="Llanowar Elves",
            card_type="Creature",
            oracle_text="{T}: Add {G}.",
            instance_id=1,
            keywords=[],
        )
        mana_dork.tapped = False
        game_state.players[0].battlefield.add_card(mana_dork)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=mana_dork,
            additional_choices={"ability_text": "{T}: Add {G}."},
        )

        validator = ActivateManaAbilityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_not_mana_ability(self):
        """Test activating non-mana ability as mana ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        creature = Card(
            name="Prodigal Pyromancer",
            card_type="Creature",
            oracle_text="{T}: Deal 1 damage to any target.",
            instance_id=1,
            keywords=[],
        )
        creature.tapped = False
        game_state.players[0].battlefield.add_card(creature)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=creature,
            additional_choices={
                "ability_text": "{T}: Deal 1 damage to any target."
            },
        )

        validator = ActivateManaAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_no_source_card(self):
        """Test mana ability activation without source card."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=None,  # No source card
        )

        validator = ActivateManaAbilityValidator()
        assert validator.validate(action, game_state) is False

    def test_valid_different_basic_lands(self):
        """Test mana ability activation for different basic lands."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        basic_lands = [
            ("Plains", "W"),
            ("Island", "U"),
            ("Swamp", "B"),
            ("Mountain", "R"),
            ("Forest", "G"),
        ]

        for i, (land_name, mana_type) in enumerate(basic_lands):
            land = Card(
                name=land_name,
                card_type="Land",
                instance_id=i + 1,
                keywords=[],
            )
            land.tapped = False
            game_state.players[0].battlefield.add_card(land)

            action = Action(
                action_type=ActionType.ACTIVATE_MANA_ABILITY,
                player_id=0,
                card=land,
            )

            validator = ActivateManaAbilityValidator()
            assert validator.validate(action, game_state) is True


class TestActivateManaAbilityExecutor:
    """Test ActivateManaAbilityExecutor."""

    def test_execute_basic_land_mana(self):
        """Test executing basic land mana ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        land = Card(
            name="Mountain",
            card_type="Land",
            instance_id=1,
            keywords=[],
        )
        land.tapped = False
        game_state.players[0].battlefield.add_card(land)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=land,
        )

        executor = ActivateManaAbilityExecutor()
        new_state = executor.execute(action, game_state)

        # Find land in new state
        new_land = None
        for card in new_state.players[0].battlefield.cards:
            if card.instance_id == 1:
                new_land = card
                break

        assert new_land is not None
        assert new_land.tapped is True  # Should be tapped

        # Check mana pool
        player = new_state.players[0]
        assert hasattr(player, "mana_pool")
        assert player.mana_pool["R"] == 1  # Mountain produces red mana

    def test_execute_different_basic_lands(self):
        """Test executing different basic land mana abilities."""
        basic_lands = [
            ("Plains", "W"),
            ("Island", "U"),
            ("Swamp", "B"),
            ("Mountain", "R"),
            ("Forest", "G"),
        ]

        for land_name, expected_mana in basic_lands:
            game_state = create_empty_game_state()
            game_state.priority_player = 0

            land = Card(
                name=land_name,
                card_type="Land",
                instance_id=1,
                keywords=[],
            )
            land.tapped = False
            game_state.players[0].battlefield.add_card(land)

            action = Action(
                action_type=ActionType.ACTIVATE_MANA_ABILITY,
                player_id=0,
                card=land,
            )

            executor = ActivateManaAbilityExecutor()
            new_state = executor.execute(action, game_state)

            player = new_state.players[0]
            assert player.mana_pool[expected_mana] == 1

    def test_execute_explicit_mana_ability(self):
        """Test executing explicit mana ability."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        mana_dork = Card(
            name="Llanowar Elves",
            card_type="Creature",
            oracle_text="{T}: Add {G}.",
            instance_id=1,
            keywords=[],
        )
        mana_dork.tapped = False
        game_state.players[0].battlefield.add_card(mana_dork)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=mana_dork,
            additional_choices={"ability_text": "{T}: Add {G}."},
        )

        executor = ActivateManaAbilityExecutor()
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

    def test_execute_multiple_mana_production(self):
        """Test executing mana ability that produces multiple mana."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0

        mana_rock = Card(
            name="Sol Ring",
            card_type="Artifact",
            oracle_text="{T}: Add {C}{C}.",
            instance_id=1,
            keywords=[],
        )
        mana_rock.tapped = False
        game_state.players[0].battlefield.add_card(mana_rock)

        action = Action(
            action_type=ActionType.ACTIVATE_MANA_ABILITY,
            player_id=0,
            card=mana_rock,
            additional_choices={"ability_text": "{T}: Add {C}{C}."},
        )

        executor = ActivateManaAbilityExecutor()
        new_state = executor.execute(action, game_state)

        # Check mana pool - should have 2 colorless mana
        player = new_state.players[0]
        assert player.mana_pool["colorless"] == 2
