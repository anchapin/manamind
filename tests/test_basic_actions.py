"""Tests for basic game action validators and executors."""

from manamind.core.action import (
    Action,
    ActionType,
    ConcedeExecutor,
    ConcedeValidator,
    DestroyExecutor,
    DestroyValidator,
    DiscardExecutor,
    DiscardValidator,
    ExileExecutor,
    ExileValidator,
    KeepHandExecutor,
    KeepHandValidator,
    MulliganExecutor,
    MulliganValidator,
    SacrificeExecutor,
    SacrificeValidator,
)
from manamind.core.game_state import Card, create_empty_game_state


class TestMulliganValidator:
    """Test MulliganValidator."""

    def test_valid_mulligan(self):
        """Test valid mulligan during pregame."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"

        # Add cards to hand
        for i in range(7):
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        validator = MulliganValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_mulligan_pregame_phase(self):
        """Test valid mulligan during pregame phase."""
        game_state = create_empty_game_state()
        game_state.phase = "pregame"

        # Add cards to hand
        for i in range(7):
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        validator = MulliganValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_mulligan_wrong_phase(self):
        """Test mulligan during wrong phase."""
        game_state = create_empty_game_state()
        game_state.phase = "main"  # Wrong phase

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        validator = MulliganValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_mulligan_no_cards(self):
        """Test mulligan with no cards in hand."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"
        # No cards in hand

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        validator = MulliganValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_mulligan_one_card(self):
        """Test mulligan with only one card in hand."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"

        # Add only one card to hand
        card = Card(name="Card", card_type="Instant")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        validator = MulliganValidator()
        assert validator.validate(action, game_state) is False


class TestMulliganExecutor:
    """Test MulliganExecutor."""

    def test_execute_mulligan(self):
        """Test executing mulligan."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"

        # Add cards to hand and library
        hand_cards = []
        for i in range(7):
            card = Card(name=f"Hand Card {i}", card_type="Instant")
            hand_cards.append(card)
            game_state.players[0].hand.add_card(card)

        for i in range(20):
            card = Card(name=f"Library Card {i}", card_type="Instant")
            game_state.players[0].library.add_card(card)

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        executor = MulliganExecutor()
        new_state = executor.execute(action, game_state)

        # Should have 6 cards in hand (7 - 1 for first mulligan)
        assert len(new_state.players[0].hand.cards) == 6

        # Should have shuffled cards back into library
        assert (
            len(new_state.players[0].library.cards) >= 20
        )  # At least original library

        # Check mulligan count in history
        mulligan_actions = [
            entry
            for entry in new_state.history
            if entry.get("action") == "mulligan" and entry.get("player") == 0
        ]
        assert len(mulligan_actions) == 1
        assert mulligan_actions[0]["mulligan_count"] == 1

    def test_execute_multiple_mulligans(self):
        """Test executing multiple mulligans reduces hand size."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"

        # Simulate player already having one mulligan by adding it to history
        game_state.history.append(
            {
                "action": "mulligan",
                "player": 0,
                "cards_drawn": 6,
                "mulligan_count": 1,
                "turn": 1,
            }
        )

        # Add cards to hand and library
        for i in range(6):  # 6 cards from previous mulligan
            card = Card(name=f"Hand Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        for i in range(20):
            card = Card(name=f"Library Card {i}", card_type="Instant")
            game_state.players[0].library.add_card(card)

        action = Action(
            action_type=ActionType.MULLIGAN,
            player_id=0,
        )

        executor = MulliganExecutor()
        new_state = executor.execute(action, game_state)

        # Should have 5 cards in hand (6 - 1 for second mulligan)
        assert len(new_state.players[0].hand.cards) == 5

        # Check mulligan count in history
        mulligan_actions = [
            entry
            for entry in new_state.history
            if entry.get("action") == "mulligan" and entry.get("player") == 0
        ]
        assert len(mulligan_actions) == 2
        assert (
            mulligan_actions[-1]["mulligan_count"] == 2
        )  # Latest mulligan should be count 2


class TestKeepHandValidator:
    """Test KeepHandValidator."""

    def test_valid_keep_hand(self):
        """Test valid keep hand during mulligan phase."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"

        # Add cards to hand
        for i in range(7):
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.KEEP_HAND,
            player_id=0,
        )

        validator = KeepHandValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_keep_hand_pregame(self):
        """Test valid keep hand during pregame phase."""
        game_state = create_empty_game_state()
        game_state.phase = "pregame"

        # Add cards to hand
        for i in range(6):  # After one mulligan
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.KEEP_HAND,
            player_id=0,
        )

        validator = KeepHandValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_keep_hand_wrong_phase(self):
        """Test keep hand during wrong phase."""
        game_state = create_empty_game_state()
        game_state.phase = "main"  # Wrong phase

        action = Action(
            action_type=ActionType.KEEP_HAND,
            player_id=0,
        )

        validator = KeepHandValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_keep_hand_no_cards(self):
        """Test keep hand with no cards."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"
        # No cards in hand

        action = Action(
            action_type=ActionType.KEEP_HAND,
            player_id=0,
        )

        validator = KeepHandValidator()
        assert validator.validate(action, game_state) is False


class TestKeepHandExecutor:
    """Test KeepHandExecutor."""

    def test_execute_keep_hand(self):
        """Test executing keep hand."""
        game_state = create_empty_game_state()
        game_state.phase = "mulligan"

        # Add cards to hand
        for i in range(7):
            card = Card(name=f"Card {i}", card_type="Instant")
            game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.KEEP_HAND,
            player_id=0,
        )

        executor = KeepHandExecutor()
        new_state = executor.execute(action, game_state)

        # Hand should remain unchanged
        assert len(new_state.players[0].hand.cards) == 7

        # Should mark that player kept their hand in history
        keep_actions = [
            entry
            for entry in new_state.history
            if entry.get("action") == "keep_hand" and entry.get("player") == 0
        ]
        assert len(keep_actions) == 1


class TestConcedeValidator:
    """Test ConcedeValidator."""

    def test_valid_concede(self):
        """Test valid concession during active game."""
        game_state = create_empty_game_state()
        # Game is not over by default

        action = Action(
            action_type=ActionType.CONCEDE,
            player_id=0,
        )

        validator = ConcedeValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_concede_game_over(self):
        """Test concession when game is already over."""
        game_state = create_empty_game_state()
        game_state.players[0].life = 0  # Game is over

        action = Action(
            action_type=ActionType.CONCEDE,
            player_id=0,
        )

        validator = ConcedeValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_concede_invalid_player(self):
        """Test concession by invalid player."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.CONCEDE,
            player_id=5,  # Invalid player ID
        )

        validator = ConcedeValidator()
        assert validator.validate(action, game_state) is False


class TestConcedeExecutor:
    """Test ConcedeExecutor."""

    def test_execute_concede(self):
        """Test executing concession."""
        game_state = create_empty_game_state()
        game_state.players[0].life = 20
        game_state.players[1].life = 15

        action = Action(
            action_type=ActionType.CONCEDE,
            player_id=0,
        )

        executor = ConcedeExecutor()
        new_state = executor.execute(action, game_state)

        # Player 0 should have 0 life (conceded)
        assert new_state.players[0].life == 0

        # Game should be over with player 1 as winner
        assert new_state.is_game_over() is True
        assert new_state.winner() == 1


class TestDiscardValidator:
    """Test DiscardValidator."""

    def test_valid_discard(self):
        """Test valid card discard."""
        game_state = create_empty_game_state()

        # Add card to hand
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.DISCARD,
            player_id=0,
            card=card,
        )

        validator = DiscardValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_discard_no_card(self):
        """Test discard without specifying card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.DISCARD,
            player_id=0,
            card=None,  # No card specified
        )

        validator = DiscardValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_discard_not_in_hand(self):
        """Test discard of card not in hand."""
        game_state = create_empty_game_state()

        # Card is not in hand
        card = Card(name="Lightning Bolt", card_type="Instant")

        action = Action(
            action_type=ActionType.DISCARD,
            player_id=0,
            card=card,
        )

        validator = DiscardValidator()
        assert validator.validate(action, game_state) is False


class TestDiscardExecutor:
    """Test DiscardExecutor."""

    def test_execute_discard(self):
        """Test executing card discard."""
        game_state = create_empty_game_state()

        # Add card to hand
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.DISCARD,
            player_id=0,
            card=card,
        )

        executor = DiscardExecutor()
        new_state = executor.execute(action, game_state)

        # Card should be in graveyard, not in hand
        assert len(new_state.players[0].hand.cards) == 0
        assert len(new_state.players[0].graveyard.cards) == 1
        assert new_state.players[0].graveyard.cards[0].name == "Lightning Bolt"
        assert new_state.players[0].graveyard.cards[0].zone == "graveyard"


class TestSacrificeValidator:
    """Test SacrificeValidator."""

    def test_valid_sacrifice(self):
        """Test valid permanent sacrifice."""
        game_state = create_empty_game_state()

        # Add permanent to battlefield
        permanent = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[0].battlefield.add_card(permanent)

        action = Action(
            action_type=ActionType.SACRIFICE,
            player_id=0,
            card=permanent,
        )

        validator = SacrificeValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_sacrifice_no_card(self):
        """Test sacrifice without specifying card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.SACRIFICE,
            player_id=0,
            card=None,  # No card specified
        )

        validator = SacrificeValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_sacrifice_not_on_battlefield(self):
        """Test sacrifice of card not on battlefield."""
        game_state = create_empty_game_state()

        # Card is in hand, not battlefield
        card = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.SACRIFICE,
            player_id=0,
            card=card,
        )

        validator = SacrificeValidator()
        assert validator.validate(action, game_state) is False


class TestSacrificeExecutor:
    """Test SacrificeExecutor."""

    def test_execute_sacrifice(self):
        """Test executing permanent sacrifice."""
        game_state = create_empty_game_state()

        # Add permanent to battlefield
        permanent = Card(name="Grizzly Bears", card_type="Creature")
        permanent.attacking = True  # Set some combat state
        permanent.tapped = True
        game_state.players[0].battlefield.add_card(permanent)

        action = Action(
            action_type=ActionType.SACRIFICE,
            player_id=0,
            card=permanent,
        )

        executor = SacrificeExecutor()
        new_state = executor.execute(action, game_state)

        # Permanent should be in graveyard, not battlefield
        assert len(new_state.players[0].battlefield.cards) == 0
        assert len(new_state.players[0].graveyard.cards) == 1
        graveyard_card = new_state.players[0].graveyard.cards[0]
        assert graveyard_card.name == "Grizzly Bears"
        assert graveyard_card.zone == "graveyard"

        # Combat states should be reset
        assert graveyard_card.attacking is False
        assert graveyard_card.tapped is False


class TestDestroyValidator:
    """Test DestroyValidator."""

    def test_valid_destroy(self):
        """Test valid permanent destruction."""
        game_state = create_empty_game_state()

        # Add permanent to battlefield
        permanent = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[0].battlefield.add_card(permanent)

        action = Action(
            action_type=ActionType.DESTROY,
            player_id=1,  # Other player destroying it
            card=permanent,
        )

        validator = DestroyValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_destroy_opponent_permanent(self):
        """Test destroying opponent's permanent."""
        game_state = create_empty_game_state()

        # Add permanent to opponent's battlefield
        permanent = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[1].battlefield.add_card(permanent)

        action = Action(
            action_type=ActionType.DESTROY,
            player_id=0,
            card=permanent,
        )

        validator = DestroyValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_destroy_no_card(self):
        """Test destruction without specifying card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.DESTROY,
            player_id=0,
            card=None,  # No card specified
        )

        validator = DestroyValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_destroy_not_permanent(self):
        """Test destruction of card not on battlefield."""
        game_state = create_empty_game_state()

        # Card is in hand, not battlefield
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.DESTROY,
            player_id=0,
            card=card,
        )

        validator = DestroyValidator()
        assert validator.validate(action, game_state) is False


class TestDestroyExecutor:
    """Test DestroyExecutor."""

    def test_execute_destroy_own_permanent(self):
        """Test executing destruction of own permanent."""
        game_state = create_empty_game_state()

        # Add permanent to battlefield
        permanent = Card(name="Grizzly Bears", card_type="Creature")
        permanent.attacking = True
        permanent.tapped = True
        game_state.players[0].battlefield.add_card(permanent)

        action = Action(
            action_type=ActionType.DESTROY,
            player_id=0,
            card=permanent,
        )

        executor = DestroyExecutor()
        new_state = executor.execute(action, game_state)

        # Permanent should be in owner's graveyard
        assert len(new_state.players[0].battlefield.cards) == 0
        assert len(new_state.players[0].graveyard.cards) == 1
        graveyard_card = new_state.players[0].graveyard.cards[0]
        assert graveyard_card.name == "Grizzly Bears"
        assert graveyard_card.zone == "graveyard"

        # Combat states should be reset
        assert graveyard_card.attacking is False
        assert graveyard_card.tapped is False

    def test_execute_destroy_opponent_permanent(self):
        """Test executing destruction of opponent's permanent."""
        game_state = create_empty_game_state()

        # Add permanent to opponent's battlefield
        permanent = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[1].battlefield.add_card(permanent)

        action = Action(
            action_type=ActionType.DESTROY,
            player_id=0,
            card=permanent,
        )

        executor = DestroyExecutor()
        new_state = executor.execute(action, game_state)

        # Permanent should be in owner's (player 1) graveyard
        assert len(new_state.players[1].battlefield.cards) == 0
        assert len(new_state.players[1].graveyard.cards) == 1
        assert new_state.players[1].graveyard.cards[0].name == "Grizzly Bears"


class TestExileValidator:
    """Test ExileValidator."""

    def test_valid_exile_from_battlefield(self):
        """Test valid exile from battlefield."""
        game_state = create_empty_game_state()

        # Add card to battlefield
        card = Card(name="Grizzly Bears", card_type="Creature")
        game_state.players[0].battlefield.add_card(card)

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        validator = ExileValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_exile_from_hand(self):
        """Test valid exile from hand."""
        game_state = create_empty_game_state()

        # Add card to hand
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        validator = ExileValidator()
        assert validator.validate(action, game_state) is True

    def test_valid_exile_from_graveyard(self):
        """Test valid exile from graveyard."""
        game_state = create_empty_game_state()

        # Add card to graveyard
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].graveyard.add_card(card)

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        validator = ExileValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_exile_no_card(self):
        """Test exile without specifying card."""
        game_state = create_empty_game_state()

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=None,  # No card specified
        )

        validator = ExileValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_exile_card_not_found(self):
        """Test exile of card not in any zone."""
        game_state = create_empty_game_state()

        # Card is not in any zone
        card = Card(name="Lightning Bolt", card_type="Instant")

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        validator = ExileValidator()
        assert validator.validate(action, game_state) is False


class TestExileExecutor:
    """Test ExileExecutor."""

    def test_execute_exile_from_battlefield(self):
        """Test executing exile from battlefield."""
        game_state = create_empty_game_state()

        # Add card to battlefield
        card = Card(name="Grizzly Bears", card_type="Creature")
        card.attacking = True
        card.tapped = True
        game_state.players[0].battlefield.add_card(card)

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        executor = ExileExecutor()
        new_state = executor.execute(action, game_state)

        # Card should be in exile, not battlefield
        assert len(new_state.players[0].battlefield.cards) == 0
        assert len(new_state.players[0].exile.cards) == 1
        exiled_card = new_state.players[0].exile.cards[0]
        assert exiled_card.name == "Grizzly Bears"
        assert exiled_card.zone == "exile"

        # Combat states should be reset
        assert exiled_card.attacking is False
        assert exiled_card.blocking is None
        assert exiled_card.tapped is False

    def test_execute_exile_from_hand(self):
        """Test executing exile from hand."""
        game_state = create_empty_game_state()

        # Add card to hand
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(card)

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        executor = ExileExecutor()
        new_state = executor.execute(action, game_state)

        # Card should be in exile, not hand
        assert len(new_state.players[0].hand.cards) == 0
        assert len(new_state.players[0].exile.cards) == 1
        assert new_state.players[0].exile.cards[0].name == "Lightning Bolt"
        assert new_state.players[0].exile.cards[0].zone == "exile"

    def test_execute_exile_from_graveyard(self):
        """Test executing exile from graveyard."""
        game_state = create_empty_game_state()

        # Add card to graveyard
        card = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].graveyard.add_card(card)

        action = Action(
            action_type=ActionType.EXILE,
            player_id=0,
            card=card,
        )

        executor = ExileExecutor()
        new_state = executor.execute(action, game_state)

        # Card should be in exile, not graveyard
        assert len(new_state.players[0].graveyard.cards) == 0
        assert len(new_state.players[0].exile.cards) == 1
        assert new_state.players[0].exile.cards[0].name == "Lightning Bolt"
        assert new_state.players[0].exile.cards[0].zone == "exile"
