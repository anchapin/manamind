# ManaMind - AI for Magic: The Gathering

## Project Overview

ManaMind is an AI agent for playing Magic: The Gathering at a superhuman level using deep reinforcement learning and self-play, inspired by AlphaZero. The project aims to create the first AI capable of playing MTG at a professional level through three phases:

1. **Phase 1**: >80% win rate against Forge AI (3-6 months)
2. **Phase 2**: Platinum rank on MTGA (6-12 months)
3. **Phase 3**: Top 100 Mythic ranking (12-24 months)

## Project Structure

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

## Key Technologies

- **ML Framework**: PyTorch with custom MTG-specific architectures
- **Game Engine**: Forge (Java-based MTG simulator) with Python-Java bridge (Py4J/Jpype1)
- **Card Data**: MTGJSON for comprehensive card information
- **Infrastructure**: Docker, Ray for distributed training
- **Development Tools**: Black, isort, flake8, mypy, pytest

## Building and Running

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

## Development Workflow

### Code Quality Checks

The project uses strict code quality standards with automated checks:

1. **MyPy Type Checking**: `mypy src`
2. **Code Formatting**: `black --check src tests`
3. **Import Sorting**: `isort --check-only src tests`
4. **Linting**: `flake8 src tests`
5. **Tests**: `pytest`

Run all checks locally with: `./scripts/local-ci-check.sh`

### Pre-commit Hooks

Set up pre-commit hooks to catch issues early:
```bash
pre-commit install
```

### Development Process

1. Make code changes
2. Run `./scripts/local-ci-check.sh`
3. Fix any issues (focus on MyPy first)
4. Repeat until all checks pass
5. Commit and push (CI will pass!)

## Architecture Overview

### Core Components

- **Game State Encoder**: Converts MTG game states to neural network inputs
- **Policy/Value Networks**: AlphaZero-style architecture for move prediction and evaluation
- **Monte Carlo Tree Search**: Guided search for optimal move selection  
- **Self-Play Training**: Primary learning mechanism through millions of games

### Training Process

The training follows the AlphaZero methodology:

1. **Self-Play Generation**: Agent plays games against itself using MCTS
2. **Data Collection**: Game positions, MCTS policies, and outcomes  
3. **Network Training**: Update policy/value networks on collected data
4. **Iteration**: Repeat with improved network

## Configuration

Main configuration is in `configs/base.yaml` with parameters for:
- Model architecture
- Training hyperparameters
- MCTS settings
- Forge integration
- Data management
- Logging

## Testing

Run tests with pytest:
```bash
# All tests
pytest

# Specific test file
pytest tests/test_game_state.py

# With coverage
pytest --cov=src/manamind
```

## CI/CD Pipeline

GitHub Actions workflow in `.github/workflows/test.yml`:
- Runs on Python 3.9, 3.10, 3.11
- Checks code quality (black, isort, flake8)
- Performs type checking (mypy)
- Runs test suite (pytest)
- Uploads coverage to Codecov

## Key Files

- `README.md`: Project overview and usage instructions
- `pyproject.toml`: Project dependencies and build configuration
- `configs/base.yaml`: Main configuration file
- `scripts/setup.sh`: Development environment setup
- `scripts/local-ci-check.sh`: Local CI validation script
- `docker/Dockerfile`: Container build configuration
- `docker/docker-compose.yml`: Multi-service orchestration

## Contributing

1. Ensure all code quality checks pass
2. Write tests for new functionality
3. Follow the existing code style
4. Update documentation as needed
5. Submit pull requests for review