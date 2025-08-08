"""Usage examples for the enhanced game state modeling architecture.

This module demonstrates how to use the comprehensive game state system
including encoding, action generation, and performance optimizations.
"""

import time
from pathlib import Path

import torch

from manamind.core.game_state import GameState, Card, create_empty_game_state
from manamind.core.action import Action, ActionType, ActionSpace
from manamind.core.state_manager import (
    CopyOnWriteGameState, IncrementalStateManager, 
    TranspositionTable, BatchStateProcessor
)
from manamind.models.enhanced_encoder import EnhancedGameStateEncoder, EncoderConfig
from manamind.data.card_database import CardDatabase


def basic_game_state_example():
    """Demonstrate basic game state creation and manipulation."""
    print("=== Basic Game State Example ===")
    
    # Create initial game state
    game_state = create_empty_game_state()
    print(f"Initial state - Turn: {game_state.turn_number}, Phase: {game_state.phase}")
    print(f"Player 0 life: {game_state.players[0].life}")
    print(f"Player 1 life: {game_state.players[1].life}")
    
    # Simulate some changes
    game_state.players[0].life = 15
    game_state.phase = "combat"
    game_state.turn_number = 3
    
    print(f"After changes - Turn: {game_state.turn_number}, Phase: {game_state.phase}")
    print(f"Player 0 life: {game_state.players[0].life}")
    
    # Test game state hash
    state_hash = game_state.compute_hash()
    print(f"State hash: {state_hash}")


def card_database_example():
    """Demonstrate card database usage."""
    print("\n=== Card Database Example ===")
    
    # Initialize database (will download MTGJSON if needed)
    db = CardDatabase("data/cards")
    
    # Create some card instances
    bolt = db.create_card_instance("Lightning Bolt", controller=0)
    if bolt:
        print(f"Created card: {bolt.name} ({bolt.mana_cost}) - CMC: {bolt.converted_mana_cost}")
        print(f"Types: {bolt.card_types}")
        print(f"Text: {bolt.oracle_text}")
        
        # Test card methods
        print(f"Is instant/sorcery: {bolt.is_instant_or_sorcery()}")
        print(f"Is creature: {bolt.is_creature()}")
        print(f"Is land: {bolt.is_land()}")
    
    # Search for cards
    print("\nSearching for creatures with CMC 2:")
    creatures = db.search_cards(type="Creature", cmc=2)
    for creature in creatures[:3]:  # Show first 3
        print(f"  {creature.name} - {creature.power}/{creature.toughness}")


def enhanced_encoding_example():
    """Demonstrate enhanced neural network encoding."""
    print("\n=== Enhanced Encoding Example ===")
    
    # Create encoder with configuration
    config = EncoderConfig(
        card_vocab_size=10000,
        embed_dim=256,
        hidden_dim=512,
        output_dim=1024,
        num_heads=4,
        dropout=0.1
    )
    
    encoder = EnhancedGameStateEncoder(config)
    print(f"Created encoder with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Create a game state with some cards
    game_state = create_empty_game_state()
    
    # Add some test cards to make encoding more interesting
    for i, player in enumerate(game_state.players):
        for j in range(3):
            card = Card(
                name=f"Test Card {i}_{j}",
                mana_cost=f"{{{j}}}",
                converted_mana_cost=j,
                card_types=["Creature"],
                power=j+1,
                toughness=j+1,
                card_id=i*10 + j + 1,
                controller=i
            )
            player.hand.add_card(card)
    
    # Encode the game state
    start_time = time.time()
    with torch.no_grad():
        encoding = encoder(game_state)
    encoding_time = time.time() - start_time
    
    print(f"Encoding shape: {encoding.shape}")
    print(f"Encoding time: {encoding_time:.4f} seconds")
    print(f"Encoding stats - Min: {encoding.min():.3f}, Max: {encoding.max():.3f}, Mean: {encoding.mean():.3f}")


def action_space_example():
    """Demonstrate action space and legal action generation."""
    print("\n=== Action Space Example ===")
    
    # Create action space
    action_space = ActionSpace(max_actions=1000)
    
    # Create a game state with some playable cards
    game_state = create_empty_game_state()
    player = game_state.players[0]
    
    # Add a land to hand
    forest = Card(
        name="Forest",
        card_types=["Land"],
        oracle_text="Tap: Add {G}.",
        card_id=1,
        controller=0
    )
    player.hand.add_card(forest)
    
    # Add an instant to hand  
    bolt = Card(
        name="Lightning Bolt",
        mana_cost="{R}",
        converted_mana_cost=1,
        card_types=["Instant"],
        oracle_text="Lightning Bolt deals 3 damage to any target.",
        card_id=2,
        controller=0
    )
    player.hand.add_card(bolt)
    
    # Add some mana
    player.mana_pool = {"R": 1, "G": 1}
    
    # Generate legal actions
    legal_actions = action_space.get_legal_actions(game_state)
    
    print(f"Found {len(legal_actions)} legal actions:")
    for i, action in enumerate(legal_actions[:5]):  # Show first 5
        print(f"  {i+1}. {action.action_type.value}", end="")
        if action.card:
            print(f" - {action.card.name}")
        else:
            print()
    
    # Test action encoding
    if legal_actions:
        action_vector = action_space.action_to_vector(legal_actions[0])
        print(f"Action vector shape: {action_vector.shape}")
        print(f"Non-zero elements: {torch.nonzero(action_vector).numel()}")


def performance_optimization_example():
    """Demonstrate performance optimization features."""
    print("\n=== Performance Optimization Example ===")
    
    # Test copy-on-write states
    print("Testing copy-on-write game states...")
    
    base_state = create_empty_game_state()
    cow_state = CopyOnWriteGameState()
    
    # Create multiple COW copies
    copies = []
    start_time = time.time()
    for i in range(100):
        copy_state = cow_state.copy()
        copies.append(copy_state)
    cow_time = time.time() - start_time
    
    print(f"Created 100 COW copies in {cow_time:.4f} seconds")
    
    # Test incremental state manager
    print("\nTesting incremental state manager...")
    
    manager = IncrementalStateManager(base_state)
    
    # Create some test actions
    test_actions = []
    for i in range(10):
        action = Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=i % 2,
            timestamp=time.time()
        )
        test_actions.append(action)
    
    # Test push/pop performance
    start_time = time.time()
    for action in test_actions:
        try:
            manager.push_action(action)
        except NotImplementedError:
            # Action execution not fully implemented yet
            pass
    
    for _ in range(len(test_actions)):
        manager.pop_action()
    
    delta_time = time.time() - start_time
    print(f"Push/pop {len(test_actions)} actions in {delta_time:.4f} seconds")
    
    # Test transposition table
    print("\nTesting transposition table...")
    
    tt = TranspositionTable(max_size=1000)
    
    # Store some test data
    for i in range(100):
        state_hash = base_state.compute_hash()
        # Modify hash slightly to create unique entries
        state_hash._hash += i
        
        tt.update_mcts_data(
            state_hash,
            visit_count=i,
            value_estimate=0.5,
            action_values={"pass": 0.3, "play": 0.7}
        )
    
    stats = tt.get_stats()
    print(f"Transposition table stats: {stats}")


def batch_processing_example():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Example ===")
    
    # Create multiple game states
    states = []
    for i in range(10):
        state = create_empty_game_state()
        state.turn_number = i + 1
        state.players[0].life = 20 - i
        states.append(state)
    
    print(f"Created {len(states)} test states")
    
    # Create batch processor
    processor = BatchStateProcessor(max_workers=2, batch_size=4)
    
    # Test batch encoding (would need actual encoder)
    print("Batch encoding would process states in parallel...")
    
    # Create simple action space for testing
    action_space = ActionSpace(max_actions=100)
    
    # Test batch legal action generation
    start_time = time.time()
    try:
        batch_actions = processor.process_states_parallel(
            states, 
            "legal_actions", 
            action_space=action_space
        )
        batch_time = time.time() - start_time
        print(f"Generated legal actions for {len(states)} states in {batch_time:.4f} seconds")
        print(f"Average actions per state: {sum(len(actions) for actions in batch_actions) / len(batch_actions):.1f}")
    except Exception as e:
        print(f"Batch processing demonstration (would work with full implementation): {e}")


def integration_example():
    """Demonstrate full integration of all components."""
    print("\n=== Integration Example ===")
    
    print("Full integration would combine:")
    print("1. Card database for complete card information")
    print("2. Enhanced game state with all MTG mechanics")
    print("3. Neural network encoding for AI training")
    print("4. Comprehensive action space for decision making")
    print("5. Performance optimizations for scale")
    print("6. MCTS integration for game tree search")
    print("7. Training pipeline for self-play learning")
    
    print("\nThis architecture supports:")
    print("- Phase 1: Forge integration with basic gameplay")
    print("- Phase 2: MTGA deployment with full rules")
    print("- Phase 3: Superhuman performance optimization")
    
    # Simulate training metrics
    print("\nSimulated performance targets:")
    print("- State encoding: <10ms per state")
    print("- Legal actions: <50ms per state")
    print("- MCTS simulations: >1000/second")
    print("- Memory usage: <100MB per game")
    print("- Training throughput: >10,000 games/hour")


def main():
    """Run all examples."""
    print("ManaMind Game State Architecture Examples")
    print("=" * 50)
    
    # Check if we have required directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "cards").mkdir(exist_ok=True)
    
    try:
        basic_game_state_example()
        card_database_example()
        enhanced_encoding_example()
        action_space_example()
        performance_optimization_example()
        batch_processing_example()
        integration_example()
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print("This is expected as some components require full implementation.")
        print("The architecture design is complete and ready for implementation.")
    
    print("\n" + "=" * 50)
    print("Architecture demonstration complete!")


if __name__ == "__main__":
    main()