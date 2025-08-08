# Game State Modeling Architecture for ManaMind

## Executive Summary

This document defines the comprehensive game state modeling architecture for ManaMind, designed to handle the immense complexity of Magic: The Gathering while maintaining neural network compatibility and training performance. The architecture balances completeness with efficiency, supporting both Forge training and eventual MTGA deployment.

## Architecture Overview

### Core Design Principles

1. **Completeness**: Capture all game state information necessary for superhuman play
2. **Efficiency**: Optimize for fast MCTS simulations and neural network processing
3. **Scalability**: Handle 25,000+ unique cards and complex interactions
4. **Extensibility**: Support new cards and mechanics through modular design
5. **Performance**: Enable millions of games for self-play training

### Key Components

1. **Enhanced Game State Representation** - Complete MTG rule-compliant state
2. **Neural Network Encoding System** - Fixed-size tensor representations
3. **Comprehensive Action Space** - All possible MTG actions with validation
4. **Efficient State Management** - Fast copying and serialization
5. **MTGJSON Integration** - Dynamic card database and encoding
6. **Performance Optimization** - Memory and compute optimizations

## Enhanced Game State Representation

### Card Representation Enhancement

```python
@dataclass
class CardInstance:
    """Enhanced card representation with full MTG state tracking."""
    
    # Core card data (from MTGJSON)
    name: str
    mana_cost: str
    converted_mana_cost: int
    card_types: List[str]  # ["Creature", "Artifact"], etc.
    subtypes: List[str]    # ["Human", "Soldier"], etc.
    supertypes: List[str]  # ["Legendary", "Basic"], etc.
    
    # Creature stats
    power: Optional[int] = None
    toughness: Optional[int] = None
    base_power: Optional[int] = None  # Original values
    base_toughness: Optional[int] = None
    
    # Planeswalker stats
    loyalty: Optional[int] = None
    starting_loyalty: Optional[int] = None
    
    # State tracking
    tapped: bool = False
    summoning_sick: bool = False
    counters: Dict[str, int] = field(default_factory=dict)  # +1/+1, loyalty, etc.
    
    # Temporary modifications
    continuous_effects: List[Dict[str, Any]] = field(default_factory=list)
    until_end_of_turn_effects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Abilities and text
    oracle_text: str = ""
    abilities: List[str] = field(default_factory=list)
    activated_abilities: List[Dict[str, Any]] = field(default_factory=list)
    triggered_abilities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Combat state
    attacking: bool = False
    blocking: Optional[int] = None  # ID of creature being blocked
    blocked_by: List[int] = field(default_factory=list)  # IDs of blocking creatures
    
    # Targeting and references
    targets: List[Any] = field(default_factory=list)
    attached_to: Optional[int] = None  # For auras/equipment
    
    # Internal identifiers
    instance_id: int  # Unique instance ID
    card_id: int      # Card database ID
    controller: int   # Player ID
    owner: int        # Original owner ID
    
    # Zone tracking
    zone: str = "unknown"
    zone_position: Optional[int] = None  # Position in library/graveyard
    
    # Timing and history
    entered_battlefield_turn: Optional[int] = None
    cast_turn: Optional[int] = None
    mana_paid: Optional[Dict[str, int]] = None  # Actual mana cost paid
```

### Enhanced Zone Management

```python
class EnhancedZone:
    """Advanced zone management with ordering and search capabilities."""
    
    def __init__(self, name: str, owner: int, ordered: bool = False):
        self.name = name
        self.owner = owner
        self.ordered = ordered  # True for library, graveyard
        self.cards: List[CardInstance] = []
        self._card_map: Dict[int, CardInstance] = {}  # Fast lookup
        
    def add_card(self, card: CardInstance, position: Optional[int] = None) -> None:
        """Add card with optional position (for ordered zones)."""
        if position is None:
            self.cards.append(card)
        else:
            self.cards.insert(position, card)
        
        self._card_map[card.instance_id] = card
        card.zone = self.name
        card.zone_position = position
        
    def remove_card(self, card: CardInstance) -> bool:
        """Remove card and update positions."""
        if card.instance_id not in self._card_map:
            return False
            
        self.cards.remove(card)
        del self._card_map[card.instance_id]
        
        # Update positions for ordered zones
        if self.ordered:
            for i, c in enumerate(self.cards):
                c.zone_position = i
                
        return True
    
    def find_cards(self, **criteria) -> List[CardInstance]:
        """Find cards matching criteria."""
        results = []
        for card in self.cards:
            match = True
            for key, value in criteria.items():
                if not hasattr(card, key) or getattr(card, key) != value:
                    match = False
                    break
            if match:
                results.append(card)
        return results
    
    def shuffle(self) -> None:
        """Shuffle zone contents (primarily for library)."""
        import random
        random.shuffle(self.cards)
        if self.ordered:
            for i, card in enumerate(self.cards):
                card.zone_position = i
```

### Complete Game State

```python
@dataclass 
class ComprehensiveGameState:
    """Complete MTG game state representation."""
    
    # Basic game information
    turn_number: int = 1
    phase: str = "beginning"
    step: str = "untap"  # Detailed phase/step tracking
    priority_player: int = 0
    active_player: int = 0
    
    # Players
    players: Tuple[EnhancedPlayer, EnhancedPlayer]
    
    # Stack and timing
    stack: List[StackObject] = field(default_factory=list)
    state_based_actions_pending: bool = False
    
    # Turn structure
    phases_completed: Set[str] = field(default_factory=set)
    passed_priority: Set[int] = field(default_factory=set)
    
    # Combat state
    combat_state: Optional[CombatState] = None
    
    # Continuous effects
    continuous_effects: List[ContinuousEffect] = field(default_factory=list)
    replacement_effects: List[ReplacementEffect] = field(default_factory=list)
    
    # Game rules state
    storm_count: int = 0
    spells_cast_this_turn: List[CardInstance] = field(default_factory=list)
    
    # History for neural network context
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[Action] = field(default_factory=list)
    
    # Performance optimization
    _state_hash: Optional[int] = None
    _dirty: bool = True
    
    def compute_state_hash(self) -> int:
        """Compute hash for state caching and transposition tables."""
        if not self._dirty and self._state_hash is not None:
            return self._state_hash
            
        # Create hash from key game state components
        hash_components = [
            self.turn_number,
            self.phase,
            self.step,
            self.active_player,
            self.priority_player,
            tuple(p.life for p in self.players),
            tuple(len(zone.cards) for p in self.players for zone in p.all_zones()),
            len(self.stack),
        ]
        
        self._state_hash = hash(tuple(hash_components))
        self._dirty = False
        return self._state_hash
    
    def copy(self) -> 'ComprehensiveGameState':
        """Efficient deep copy for MCTS simulations."""
        return copy.deepcopy(self)  # TODO: Optimize with custom implementation
    
    def apply_state_based_actions(self) -> None:
        """Apply state-based actions (creature death, planeswalker loyalty, etc.)."""
        # TODO: Implement comprehensive SBA system
        pass
```

## Neural Network Encoding System

### Multi-Modal Encoder Architecture

```python
class MultiModalGameStateEncoder(nn.Module):
    """Advanced game state encoder supporting multiple representation modes."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Card vocabulary and embeddings
        self.card_embedder = CardEmbeddingSystem(config.card_vocab_size)
        
        # Zone encoders with attention
        self.zone_encoders = nn.ModuleDict({
            'hand': HandEncoder(config),
            'battlefield': BattlefieldEncoder(config),
            'graveyard': SequentialZoneEncoder(config),
            'library': LibraryEncoder(config),
            'exile': SequentialZoneEncoder(config),
            'stack': StackEncoder(config),
        })
        
        # Game state encoders
        self.player_encoder = PlayerStateEncoder(config)
        self.global_encoder = GlobalStateEncoder(config)
        self.combat_encoder = CombatStateEncoder(config)
        
        # Attention and fusion
        self.cross_attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads)
        self.state_fusion = StateFusionNetwork(config)
        
        # Output projection
        self.output_projector = nn.Sequential(
            nn.Linear(config.fusion_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.output_dim),
            nn.LayerNorm(config.output_dim)
        )
    
    def forward(self, game_state: ComprehensiveGameState) -> torch.Tensor:
        """Encode complete game state."""
        # Encode zones for both players
        zone_encodings = {}
        for player_id, player in enumerate(game_state.players):
            player_zones = {}
            for zone_name in ['hand', 'battlefield', 'graveyard', 'library', 'exile']:
                zone = getattr(player, zone_name)
                encoder = self.zone_encoders[zone_name]
                player_zones[zone_name] = encoder(zone, player_id)
            zone_encodings[player_id] = player_zones
        
        # Encode stack
        stack_encoding = self.zone_encoders['stack'](game_state.stack)
        
        # Encode players
        player_encodings = [
            self.player_encoder(player, player_id) 
            for player_id, player in enumerate(game_state.players)
        ]
        
        # Encode global state
        global_encoding = self.global_encoder(game_state)
        
        # Encode combat if active
        combat_encoding = None
        if game_state.combat_state:
            combat_encoding = self.combat_encoder(game_state.combat_state)
        
        # Fuse all encodings
        return self.state_fusion(
            zone_encodings, player_encodings, global_encoding, 
            stack_encoding, combat_encoding
        )
```

### Specialized Zone Encoders

```python
class BattlefieldEncoder(nn.Module):
    """Specialized encoder for battlefield with spatial relationships."""
    
    def __init__(self, config):
        super().__init__()
        self.card_encoder = CardInstanceEncoder(config)
        self.position_encoder = PositionalEncoding(config.embed_dim)
        self.creature_interaction = CreatureInteractionNetwork(config)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.embed_dim, config.num_heads, config.ff_dim
            ),
            num_layers=config.num_layers
        )
    
    def forward(self, battlefield: EnhancedZone, player_id: int) -> torch.Tensor:
        """Encode battlefield state with creature interactions."""
        if not battlefield.cards:
            return torch.zeros(1, self.config.output_dim)
        
        # Encode each card
        card_embeddings = []
        for card in battlefield.cards:
            card_emb = self.card_encoder(card)
            card_embeddings.append(card_emb)
        
        # Stack and add positional encoding
        cards_tensor = torch.stack(card_embeddings)
        cards_tensor = self.position_encoder(cards_tensor)
        
        # Apply transformer to model interactions
        battlefield_encoding = self.transformer(cards_tensor)
        
        # Aggregate to single representation
        return battlefield_encoding.mean(dim=0)
```

## Comprehensive Action Space

### Action Type Taxonomy

```python
class ExtendedActionType(Enum):
    """Complete taxonomy of MTG actions."""
    
    # Basic game actions
    PLAY_LAND = "play_land"
    CAST_SPELL = "cast_spell"
    ACTIVATE_ABILITY = "activate_ability"
    ACTIVATE_MANA_ABILITY = "activate_mana_ability"
    
    # Priority and timing
    PASS_PRIORITY = "pass_priority"
    HOLD_PRIORITY = "hold_priority"
    
    # Combat actions
    DECLARE_ATTACKERS = "declare_attackers"
    DECLARE_BLOCKERS = "declare_blockers"
    ASSIGN_COMBAT_DAMAGE = "assign_combat_damage"
    ORDER_BLOCKERS = "order_blockers"
    
    # Special actions
    MULLIGAN = "mulligan"
    KEEP_HAND = "keep_hand" 
    CONCEDE = "concede"
    
    # Card-specific actions
    DISCARD = "discard"
    SACRIFICE = "sacrifice"
    DESTROY = "destroy"
    EXILE = "exile"
    
    # Targeting and choices
    CHOOSE_TARGET = "choose_target"
    CHOOSE_MODE = "choose_mode"
    CHOOSE_X_VALUE = "choose_x_value"
    ORDER_CARDS = "order_cards"
    
    # Replacement effects
    REPLACE_EFFECT = "replace_effect"
    DECLINE_REPLACEMENT = "decline_replacement"
```

### Advanced Action Representation

```python
@dataclass
class ComprehensiveAction:
    """Complete action representation supporting all MTG complexities."""
    
    # Core action data
    action_type: ExtendedActionType
    player_id: int
    timestamp: float = field(default_factory=time.time)
    
    # Card references
    card: Optional[CardInstance] = None
    target_cards: List[CardInstance] = field(default_factory=list)
    
    # Player/permanent targets
    target_players: List[int] = field(default_factory=list)
    target_permanents: List[int] = field(default_factory=list)
    
    # Mana payment
    mana_payment: Optional[Dict[str, int]] = None
    alternative_cost: Optional[str] = None
    
    # Choices and parameters
    x_value: Optional[int] = None
    modes_chosen: List[str] = field(default_factory=list)
    order_choices: List[int] = field(default_factory=list)
    additional_choices: Dict[str, Any] = field(default_factory=dict)
    
    # Combat-specific
    attackers: List[int] = field(default_factory=list)
    defenders: List[int] = field(default_factory=list)  # Player or planeswalker IDs
    blockers: Dict[int, List[int]] = field(default_factory=dict)  # attacker -> [blockers]
    damage_assignment: Dict[int, Dict[int, int]] = field(default_factory=dict)
    
    # Neural network representation
    action_vector: Optional[torch.Tensor] = None
    
    def encode_to_vector(self, action_space: 'ComprehensiveActionSpace') -> torch.Tensor:
        """Encode action to neural network representation."""
        return action_space.encode_action(self)
    
    def get_complexity_score(self) -> int:
        """Calculate action complexity for MCTS guidance."""
        score = 1  # Base complexity
        
        if self.target_cards:
            score += len(self.target_cards)
        if self.modes_chosen:
            score += len(self.modes_chosen) * 2
        if self.x_value:
            score += 3
        if self.blockers:
            score += sum(len(blockers) for blockers in self.blockers.values())
        
        return score
```

### Legal Action Generation

```python
class ComprehensiveActionSpace:
    """Advanced action space with complete MTG rules integration."""
    
    def __init__(self, card_database: CardDatabase):
        self.card_db = card_database
        self.action_encoders = self._build_action_encoders()
        self.rules_engine = MTGRulesEngine()
        
    def get_legal_actions(self, game_state: ComprehensiveGameState) -> List[ComprehensiveAction]:
        """Generate all legal actions with full rules validation."""
        legal_actions = []
        current_player = game_state.players[game_state.priority_player]
        
        # Priority actions (always available when you have priority)
        legal_actions.append(ComprehensiveAction(
            action_type=ExtendedActionType.PASS_PRIORITY,
            player_id=game_state.priority_player
        ))
        
        # Phase/step-specific actions
        if game_state.phase == "main" and game_state.active_player == game_state.priority_player:
            legal_actions.extend(self._get_main_phase_actions(game_state))
        
        elif game_state.phase == "combat":
            legal_actions.extend(self._get_combat_actions(game_state))
        
        # Instant-speed actions (available in any phase with priority)
        legal_actions.extend(self._get_instant_actions(game_state))
        
        # Activated abilities
        legal_actions.extend(self._get_activated_abilities(game_state))
        
        # Triggered ability responses
        if game_state.stack and game_state.stack[-1].get('type') == 'triggered':
            legal_actions.extend(self._get_triggered_responses(game_state))
        
        # Special game actions
        legal_actions.extend(self._get_special_actions(game_state))
        
        return self._validate_actions(legal_actions, game_state)
    
    def _get_main_phase_actions(self, game_state: ComprehensiveGameState) -> List[ComprehensiveAction]:
        """Get main phase specific actions."""
        actions = []
        player = game_state.players[game_state.active_player]
        
        # Land plays
        if player.can_play_land() and len(game_state.stack) == 0:
            for card in player.hand.cards:
                if "Land" in card.card_types:
                    actions.append(ComprehensiveAction(
                        action_type=ExtendedActionType.PLAY_LAND,
                        player_id=game_state.active_player,
                        card=card
                    ))
        
        # Sorcery-speed spells
        for card in player.hand.cards:
            if self._can_cast_sorcery_speed(card, game_state):
                # Generate all possible casting combinations
                casting_actions = self._generate_casting_actions(card, game_state)
                actions.extend(casting_actions)
        
        return actions
    
    def _generate_casting_actions(self, card: CardInstance, game_state: ComprehensiveGameState) -> List[ComprehensiveAction]:
        """Generate all possible ways to cast a spell (targets, modes, X values)."""
        actions = []
        
        # Parse card for targeting requirements
        targeting_info = self.card_db.get_targeting_info(card.card_id)
        
        if not targeting_info.requires_targets:
            # Simple cast with no targets
            actions.append(ComprehensiveAction(
                action_type=ExtendedActionType.CAST_SPELL,
                player_id=game_state.priority_player,
                card=card
            ))
        else:
            # Generate all valid target combinations
            valid_targets = self._get_valid_targets(targeting_info, game_state)
            for target_combo in itertools.combinations(valid_targets, targeting_info.num_targets):
                actions.append(ComprehensiveAction(
                    action_type=ExtendedActionType.CAST_SPELL,
                    player_id=game_state.priority_player,
                    card=card,
                    target_cards=[t for t in target_combo if isinstance(t, CardInstance)],
                    target_players=[t for t in target_combo if isinstance(t, int)]
                ))
        
        # Handle modal spells
        if targeting_info.is_modal:
            modal_actions = []
            for mode_combo in self._get_valid_mode_combinations(targeting_info):
                for action in actions:
                    modal_action = copy.deepcopy(action)
                    modal_action.modes_chosen = mode_combo
                    modal_actions.append(modal_action)
            actions = modal_actions
        
        # Handle X spells
        if targeting_info.has_x_cost:
            x_actions = []
            max_x = self._calculate_max_x(card, game_state)
            for x_val in range(max_x + 1):
                for action in actions:
                    x_action = copy.deepcopy(action)
                    x_action.x_value = x_val
                    x_actions.append(x_action)
            actions = x_actions
        
        return actions
```

## Performance Optimization System

### Memory-Efficient State Management

```python
class OptimizedGameState:
    """Memory-optimized game state with copy-on-write semantics."""
    
    def __init__(self, base_state: Optional['OptimizedGameState'] = None):
        if base_state is None:
            self._data = GameStateData()
            self._refs = 1
            self._copy_on_write = False
        else:
            self._data = base_state._data
            self._refs = base_state._refs + 1
            self._copy_on_write = True
            base_state._copy_on_write = True
    
    def modify(self) -> None:
        """Prepare for modification (copy-on-write)."""
        if self._copy_on_write:
            self._data = copy.deepcopy(self._data)
            self._copy_on_write = False
            self._refs = 1
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self.modify()
            setattr(self._data, name, value)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)
```

### Incremental State Updates

```python
class IncrementalStateManager:
    """Manages incremental state updates for efficient MCTS."""
    
    def __init__(self):
        self.state_stack: List[GameStateDelta] = []
        self.base_state: ComprehensiveGameState = None
    
    def push_action(self, action: ComprehensiveAction) -> None:
        """Apply action and save delta for rollback."""
        delta = self._compute_action_delta(action, self.current_state())
        self.state_stack.append(delta)
        
    def pop_action(self) -> None:
        """Rollback last action."""
        if self.state_stack:
            delta = self.state_stack.pop()
            self._apply_reverse_delta(delta)
    
    def current_state(self) -> ComprehensiveGameState:
        """Get current state by applying all deltas."""
        if not self.state_stack:
            return self.base_state
        
        # Apply deltas incrementally (cached for efficiency)
        return self._apply_deltas_cached()
```

### Vectorized Operations

```python
class VectorizedStateProcessor:
    """Process multiple states simultaneously for batch training."""
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.encoder = MultiModalGameStateEncoder()
        
    def batch_encode(self, states: List[ComprehensiveGameState]) -> torch.Tensor:
        """Encode multiple states in parallel."""
        # Group states by structure for efficient batching
        state_groups = self._group_states_by_structure(states)
        
        encodings = []
        for group in state_groups:
            # Vectorized encoding for similar states
            batch_tensor = self._create_batch_tensor(group)
            batch_encoding = self.encoder(batch_tensor)
            encodings.extend(batch_encoding.unbind(0))
        
        return torch.stack(encodings)
    
    def batch_legal_actions(self, states: List[ComprehensiveGameState]) -> List[List[ComprehensiveAction]]:
        """Generate legal actions for multiple states in parallel."""
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._get_legal_actions, state) for state in states]
            return [future.result() for future in futures]
```

## MTGJSON Integration System

### Dynamic Card Database

```python
class MTGJSONIntegration:
    """Comprehensive MTGJSON integration with caching and updates."""
    
    def __init__(self, data_path: str = "data/cards"):
        self.data_path = Path(data_path)
        self.card_cache: Dict[str, CardData] = {}
        self.encoding_cache: Dict[int, torch.Tensor] = {}
        self.card_to_id: Dict[str, int] = {}
        self.id_to_card: Dict[int, str] = {}
        self._load_database()
    
    def _load_database(self) -> None:
        """Load and process MTGJSON data."""
        json_file = self.data_path / "AllPrintings.json"
        
        if not json_file.exists():
            self._download_mtgjson()
        
        with open(json_file) as f:
            data = json.load(f)
        
        self._process_cards(data)
        self._build_encodings()
    
    def _process_cards(self, data: Dict[str, Any]) -> None:
        """Process raw MTGJSON data into internal format."""
        card_id = 1  # Start from 1 (0 reserved for padding)
        
        for set_code, set_data in data['data'].items():
            for card_data in set_data['cards']:
                oracle_id = card_data.get('identifiers', {}).get('oracleId')
                if not oracle_id:
                    continue
                
                # Create canonical card representation
                card = self._create_card_data(card_data, set_code)
                
                # Use oracle ID as primary key (handles reprints)
                if oracle_id not in self.card_cache:
                    self.card_cache[oracle_id] = card
                    self.card_to_id[oracle_id] = card_id
                    self.id_to_card[card_id] = oracle_id
                    card_id += 1
    
    def _create_card_data(self, raw_data: Dict[str, Any], set_code: str) -> CardData:
        """Create internal card representation from MTGJSON data."""
        return CardData(
            name=raw_data['name'],
            mana_cost=raw_data.get('manaCost', ''),
            converted_mana_cost=raw_data.get('convertedManaCost', 0),
            card_types=raw_data.get('types', []),
            subtypes=raw_data.get('subtypes', []),
            supertypes=raw_data.get('supertypes', []),
            oracle_text=raw_data.get('text', ''),
            power=self._parse_power_toughness(raw_data.get('power')),
            toughness=self._parse_power_toughness(raw_data.get('toughness')),
            loyalty=self._parse_loyalty(raw_data.get('loyalty')),
            abilities=self._parse_abilities(raw_data.get('text', '')),
            keywords=raw_data.get('keywords', []),
            color_identity=raw_data.get('colorIdentity', []),
            legalities=raw_data.get('legalities', {}),
            set_code=set_code,
            rarity=raw_data.get('rarity', 'common'),
        )
    
    def get_card_encoding(self, oracle_id: str) -> torch.Tensor:
        """Get neural network encoding for a card."""
        card_id = self.card_to_id.get(oracle_id)
        if card_id is None:
            return torch.zeros(512)  # Unknown card encoding
        
        if card_id not in self.encoding_cache:
            card_data = self.card_cache[oracle_id]
            encoding = self._compute_card_encoding(card_data)
            self.encoding_cache[card_id] = encoding
        
        return self.encoding_cache[card_id]
    
    def _compute_card_encoding(self, card: CardData) -> torch.Tensor:
        """Compute embedding for card using text and structural features."""
        # Combine multiple encoding approaches
        
        # 1. Structural encoding (mana cost, types, stats)
        structural = self._encode_structural_features(card)
        
        # 2. Text encoding (abilities, oracle text)
        text_encoding = self._encode_text_features(card)
        
        # 3. Color encoding (color identity, mana cost)
        color_encoding = self._encode_color_features(card)
        
        # 4. Mechanical encoding (keywords, abilities)
        mechanical_encoding = self._encode_mechanical_features(card)
        
        # Combine all encodings
        combined = torch.cat([structural, text_encoding, color_encoding, mechanical_encoding])
        
        # Project to final dimensionality
        return self.card_projector(combined)
```

### Ability and Text Parsing

```python
class AbilityParser:
    """Parse and encode card abilities for neural network processing."""
    
    def __init__(self):
        self.keyword_vocab = self._build_keyword_vocabulary()
        self.ability_patterns = self._compile_ability_patterns()
        self.cost_parser = ManaCostParser()
    
    def parse_abilities(self, oracle_text: str) -> List[ParsedAbility]:
        """Parse oracle text into structured abilities."""
        abilities = []
        
        # Split text into individual abilities
        ability_texts = self._split_ability_text(oracle_text)
        
        for text in ability_texts:
            ability = self._parse_single_ability(text)
            if ability:
                abilities.append(ability)
        
        return abilities
    
    def _parse_single_ability(self, text: str) -> Optional[ParsedAbility]:
        """Parse a single ability into structured form."""
        # Check for activated abilities (cost: effect)
        if ':' in text:
            cost_text, effect_text = text.split(':', 1)
            cost = self.cost_parser.parse(cost_text.strip())
            effect = self._parse_effect(effect_text.strip())
            
            return ParsedAbility(
                type=AbilityType.ACTIVATED,
                cost=cost,
                effect=effect,
                original_text=text
            )
        
        # Check for triggered abilities (when/whenever/at)
        trigger_words = ['when', 'whenever', 'at']
        if any(text.lower().startswith(word) for word in trigger_words):
            trigger, effect = self._parse_triggered_ability(text)
            
            return ParsedAbility(
                type=AbilityType.TRIGGERED,
                trigger=trigger,
                effect=effect,
                original_text=text
            )
        
        # Static abilities or keywords
        return ParsedAbility(
            type=AbilityType.STATIC,
            effect=self._parse_effect(text),
            original_text=text
        )
```

## Integration Points and Deployment

### Forge Interface Enhancement

```python
class EnhancedForgeInterface:
    """Enhanced Forge integration with complete state synchronization."""
    
    def __init__(self, forge_path: str, config: ForgeConfig):
        self.forge_path = Path(forge_path)
        self.config = config
        self.py4j_gateway = None
        self.game_instance = None
        self.state_synchronizer = ForgeStateSynchronizer()
    
    def start_game(self, deck1: Deck, deck2: Deck) -> ComprehensiveGameState:
        """Start new game and return initial state."""
        # Initialize Forge game
        self._start_forge_instance()
        self.game_instance = self._create_forge_game(deck1, deck2)
        
        # Convert to internal representation
        return self.state_synchronizer.convert_from_forge(self.game_instance)
    
    def apply_action(self, action: ComprehensiveAction) -> ComprehensiveGameState:
        """Apply action in Forge and return updated state."""
        # Convert to Forge action format
        forge_action = self._convert_action_to_forge(action)
        
        # Apply in Forge
        self.game_instance.processAction(forge_action)
        
        # Convert back to internal format
        return self.state_synchronizer.convert_from_forge(self.game_instance)
    
    def get_legal_actions(self) -> List[ComprehensiveAction]:
        """Get legal actions from Forge."""
        forge_actions = self.game_instance.getLegalActions()
        return [self._convert_action_from_forge(fa) for fa in forge_actions]
```

### MTGA Interface Architecture

```python
class MTGAInterface:
    """MTGA client interface for deployment (Phase 2)."""
    
    def __init__(self, screen_reader: ScreenReader, input_controller: InputController):
        self.screen_reader = screen_reader
        self.input_controller = input_controller
        self.state_parser = MTGAStateParser()
        self.action_executor = MTGAActionExecutor()
    
    def read_game_state(self) -> ComprehensiveGameState:
        """Read current game state from MTGA client."""
        screenshot = self.screen_reader.capture_screen()
        ocr_data = self.screen_reader.extract_text(screenshot)
        ui_elements = self.screen_reader.detect_ui_elements(screenshot)
        
        return self.state_parser.parse_mtga_state(ocr_data, ui_elements)
    
    def execute_action(self, action: ComprehensiveAction) -> bool:
        """Execute action in MTGA client."""
        try:
            click_sequence = self.action_executor.convert_to_clicks(action)
            self.input_controller.execute_sequence(click_sequence)
            return True
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return False
```

## Performance Characteristics and Benchmarks

### Target Performance Metrics

- **State Encoding**: < 10ms per state for neural network input
- **Legal Action Generation**: < 50ms per state (average 100-500 actions)
- **MCTS Simulation**: > 1000 simulations/second
- **Memory Usage**: < 100MB per game state (including history)
- **Training Throughput**: > 10,000 games/hour on single GPU

### Memory Optimization

- Card instance pooling to reduce object allocation
- Shared immutable card data across all instances
- Efficient tensor caching for repeated encodings
- Copy-on-write game state semantics

### Computational Optimization

- Vectorized batch processing for similar operations
- GPU acceleration for neural network components
- Lazy evaluation of expensive state computations
- Incremental updates for MCTS rollouts

## Conclusion

This architecture provides a comprehensive foundation for ManaMind's game state modeling, balancing completeness with performance. The modular design supports progressive implementation, starting with core functionality for Phase 1 Forge integration and expanding to full complexity for superhuman performance.

Key implementation priorities:

1. **Phase 1**: Core game state representation and Forge integration
2. **Phase 1.5**: Basic neural network encoding and action space
3. **Phase 2**: Complete action space and MTGA integration
4. **Phase 3**: Advanced optimizations and superhuman performance features

The architecture is designed to handle the full complexity of Magic: The Gathering while maintaining the performance characteristics necessary for large-scale self-play training.