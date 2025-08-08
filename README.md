# ManaMind

An AI agent for playing Magic: The Gathering at superhuman level using deep reinforcement learning and self-play, inspired by AlphaZero.

## 🎯 Project Vision

ManaMind aims to create the first AI agent capable of playing Magic: The Gathering at a superhuman level, progressing through three ambitious phases:

- **Phase 1** (3-6 months): >80% win rate against Forge AI
- **Phase 2** (6-12 months): Platinum rank on MTG Arena  
- **Phase 3** (12-24 months): Top 100 Mythic ranking

## 🏗️ Project Structure

```
manamind/
├── src/manamind/              # Main source code
│   ├── core/                  # Core game logic
│   │   ├── game_state.py     # Game state representation & encoding
│   │   ├── action.py         # Action system & validation
│   │   └── agent.py          # Agent interfaces & MCTS
│   ├── models/               # Neural network architectures
│   │   ├── policy_value_network.py  # Main AlphaZero-style network
│   │   └── components.py     # Reusable NN components
│   ├── forge_interface/      # Forge game engine integration
│   │   ├── forge_client.py   # Python-Java bridge
│   │   ├── game_runner.py    # Game execution
│   │   └── state_parser.py   # State parsing
│   ├── training/            # Training infrastructure
│   │   ├── self_play.py     # Self-play training loop
│   │   ├── neural_trainer.py # Network training
│   │   └── data_manager.py  # Training data management
│   ├── evaluation/          # Model evaluation
│   ├── utils/              # Utilities
│   └── cli/                # Command line interface
├── configs/                # Configuration files
├── tests/                 # Test suite
├── docker/               # Docker configurations
├── scripts/             # Development scripts
└── data/               # Data directories
    ├── checkpoints/   # Model checkpoints
    ├── logs/         # Training logs
    ├── game_logs/    # Game data
    └── cards/        # Card database
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Java 11+ (for Forge integration)
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd manamind
   ./scripts/setup.sh
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Test Forge integration:**
   ```bash
   manamind forge-test
   ```

4. **Start training:**
   ```bash
   manamind train --config configs/base.yaml
   ```

### Using Docker

For containerized development and training:

```bash
# Development environment with Jupyter
docker-compose --profile development up

# Training
docker-compose --profile training up

# Distributed training
docker-compose --profile distributed up
```

## 🧠 Architecture Overview

### Core Components

- **Game State Encoder**: Converts MTG game states to neural network inputs
- **Policy/Value Networks**: AlphaZero-style architecture for move prediction and evaluation
- **Monte Carlo Tree Search**: Guided search for optimal move selection  
- **Self-Play Training**: Primary learning mechanism through millions of games

### Key Technologies

- **Training Environment**: Forge game engine (Java-based MTG simulator)
- **ML Framework**: PyTorch with custom MTG-specific architectures
- **Card Data**: MTGJSON for comprehensive card information
- **Infrastructure**: Docker, Ray for distributed training

## 📊 Training Process

The training follows the AlphaZero methodology:

1. **Self-Play Generation**: Agent plays games against itself using MCTS
2. **Data Collection**: Game positions, MCTS policies, and outcomes  
3. **Network Training**: Update policy/value networks on collected data
4. **Iteration**: Repeat with improved network

### Training Configuration

Key parameters (configurable in `configs/base.yaml`):

```yaml
training:
  games_per_iteration: 100     # Self-play games per training iteration
  mcts_simulations: 800        # MCTS simulations per move
  training_iterations: 1000    # Total training iterations
  batch_size: 64              # Neural network batch size
```

## 🎮 Usage Examples

### Training

```bash
# Basic training
manamind train

# Custom configuration
manamind train --config configs/phase1.yaml --iterations 500

# Resume from checkpoint
manamind train --resume checkpoints/latest.pt
```

### Evaluation

```bash
# Evaluate against Forge AI
manamind eval model.pt --opponent forge --games 50

# Evaluate against random opponent
manamind eval model.pt --opponent random --games 100
```

### System Information

```bash
# Check installation and system info
manamind info
```

## 🔧 Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_game_state.py

# With coverage
pytest --cov=src/manamind
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/manamind

# Linting
flake8 src/ tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## 📋 Development Phases

### Phase 1: Foundation & Forge Integration (Current)

**Goal**: >80% win rate against built-in Forge AI

**Key Milestones**:
- ✅ Project scaffolding and architecture
- 🔄 Python-Java bridge for Forge communication
- 🔄 Basic game state encoding
- ⏳ Initial self-play training loop
- ⏳ MCTS implementation with neural network guidance

### Phase 2: Mastery & MTGA Adaptation

**Goal**: Platinum rank on MTGA ladder

**Key Milestones**:
- Scale self-play infrastructure
- Develop MTGA screen reading interface  
- Achieve expert-level performance in Forge
- Deploy to MTGA for live evaluation

### Phase 3: Superhuman Performance

**Goal**: Top 100 Mythic ranking

**Key Milestones**:
- Consistent Mythic-level play
- Novel deck generation experiments
- Exhibition matches vs human experts

## 🧪 Technical Challenges

### Addressed

- **Complex State Space**: Custom neural network architectures for MTG
- **Variable Action Space**: Dynamic action generation and encoding
- **Hidden Information**: MCTS adapted for imperfect information games

### In Progress  

- **Forge Integration**: Building reliable Python-Java communication
- **Training Scale**: Optimizing for millions of self-play games
- **Memory Efficiency**: Handling large training datasets

### Future Challenges

- **MTGA Integration**: Screen reading without official API
- **Meta Adaptation**: Continuous learning as new cards release
- **Human-Level Strategy**: Discovering novel gameplay patterns

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepMind**: AlphaZero methodology and inspiration
- **Forge Project**: Open-source MTG rules engine
- **MTGJSON**: Comprehensive card database
- **MTG Community**: Inspiration and domain expertise

## 📞 Contact

- Project Lead: [Your Name]
- Email: team@manamind.ai
- Discord: [Server Invite]
- Issues: [GitHub Issues]

---

*ManaMind - Bringing superhuman AI to the multiverse of Magic: The Gathering*