# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ManaMind is an AI agent designed to play Magic: The Gathering (MTG) at a superhuman level, using deep reinforcement learning and self-play similar to AlphaZero. The project integrates with the Forge game engine for training and eventually with MTG Arena for live play.

## Development Architecture

### Core Components (As Planned)
- **Game State Encoder**: Neural network converting MTG game states to numerical tensors
- **Policy/Value Networks**: AlphaZero-style architecture for move prediction and position evaluation  
- **Monte Carlo Tree Search (MCTS)**: Guided search for optimal move selection
- **Self-Play Training Loop**: Primary learning mechanism using millions of games
- **Forge Interface**: Python-Java bridge for communicating with Forge game engine
- **MTGA Interface**: Screen reading/input simulation for live Arena play

### Key Technologies
- **Primary Training Environment**: Forge ([github.com/Card-Forge/forge](https://github.com/Card-Forge/forge)) - Java-based MTG rules engine
- **Card Data Source**: MTGJSON ([mtgjson.com](https://mtgjson.com/)) for canonical card information
- **ML Framework**: Python-based (specific framework TBD - likely PyTorch or TensorFlow)
- **Infrastructure**: Dockerized distributed training system

## Development Phases

### Phase 1: Foundation & Forge Integration (3-6 months)
**Goal**: 80% win rate against built-in Forge AI
- Build Python-Java interface for Forge communication
- Implement basic game state encoding
- Create initial RL training loop
- Establish headless Forge operation for parallel training

### Phase 2: Mastery & MTGA Adaptation (6-12 months) 
**Goal**: Platinum rank on MTGA ladder
- Scale self-play infrastructure massively
- Develop screen reading/input simulation for MTGA
- Optimize training for expert-level performance

### Phase 3: Superhuman Performance (12-24 months)
**Goal**: Top 100 Mythic ranking
- Push for consistent Mythic-level play
- Explore novel deck generation
- Organize exhibition matches vs human experts

## Critical Risks & Considerations

### Technical Challenges
- **Forge Integration Complexity**: Primary technical risk is building stable Python-Java interface
- **MTGA Client Brittleness**: No official API; screen reading will break with updates
- **Computational Requirements**: Training will require significant cloud resources
- **State Space Complexity**: MTG has immense branching factor and hidden information

### Security & Compliance
- MTGA integration must comply with Terms of Service
- Avoid any automation that could be considered cheating during development
- Keep training focused on Forge environment initially

## Development Commands

### Code Quality & Formatting (CRITICAL)
Always run these commands before committing to prevent CI failures:
```bash
# Format code with black (79 character limit to match CI)
black --line-length 79 src tests

# Sort imports
isort src tests

# Check for linting issues 
flake8 src tests

# Type checking
mypy src

# Run tests with coverage
pytest --cov=src/manamind --cov-report=xml --cov-report=term-missing -v

# Run all quality checks at once (recommended before commit)
black --line-length 79 src tests && isort src tests && flake8 src tests && mypy src
```

### Training Commands (Future)
```bash
# Training commands (future)
python train.py --config configs/base.yaml
python eval.py --model checkpoints/latest.pt --opponent forge_ai

# Forge integration (future)  
python forge_interface.py --test-connection
python run_forge_games.py --num-games 1000

# MTGA interface (future)
python mtga_interface.py --test-mode
```

## Key Files to Monitor

- `PRD_ProjectManaMind.md` - Complete project requirements and roadmap
- Future: `forge_interface/` - Critical Python-Java bridge code
- Future: `models/` - Neural network architectures
- Future: `training/` - Self-play and learning loops
- Future: `mtga_interface/` - Arena client integration

## Development Guidelines

### Code Quality Standards (MANDATORY)
**CRITICAL: Always follow these practices to prevent CI failures and PR delays:**

1. **Pre-commit Code Quality Checks:**
   - Run `black --line-length 79 src tests` before every commit
   - Run `isort src tests` to fix import ordering  
   - Run `flake8 src tests` and fix ALL violations before committing
   - Run `mypy src` and address type issues
   - Use the one-liner: `black --line-length 79 src tests && isort src tests && flake8 src tests && mypy src`

2. **Import Management:**
   - Remove unused imports immediately when refactoring
   - Use isort to maintain consistent import ordering
   - Avoid importing modules that aren't used

3. **Line Length & Formatting:**
   - Maintain 79-character line limit (matches CI configuration)  
   - Let black handle most formatting automatically
   - Break long lines in function arguments, not comments
   - Use meaningful variable names even if they're longer

4. **Code Writing Best Practices:**
   - Write code that passes linting from the start
   - Fix linting issues as you code, not after
   - Use type hints consistently
   - Remove debug prints and unused variables immediately

### Automation Recommendations
To reduce PR iteration cycles, consider setting up:

1. **Pre-commit hooks** (add to `.pre-commit-config.yaml`):
   ```yaml
   repos:
   - repo: https://github.com/psf/black
     rev: 25.1.0
     hooks:
     - id: black
       args: [--line-length=79]
   - repo: https://github.com/PyCQA/isort
     rev: 5.13.2
     hooks:
     - id: isort
   - repo: https://github.com/PyCQA/flake8
     rev: 7.3.0
     hooks:
     - id: flake8
   ```

2. **IDE/Editor Integration:**
   - Configure VS Code/PyCharm to run black on save
   - Enable flake8 linting in your editor
   - Set up mypy type checking in your IDE

3. **Git Aliases** (add to ~/.gitconfig):
   ```bash
   [alias]
       lint = !black --line-length 79 src tests && isort src tests && flake8 src tests && mypy src
       commit-clean = !git add . && git lint && git commit
   ```

### Code Organization
- Separate training environment (Forge) from deployment environment (MTGA)
- Modular design allowing different RL algorithms to be swapped
- Clear separation between game logic, AI logic, and interface layers

### Testing Strategy  
- Unit tests for all game state encoding/decoding
- Integration tests for Forge communication
- Performance benchmarks for training throughput
- Validation games against known-good opponents

### Performance Priorities
1. Training speed optimization (games per second)
2. Memory efficiency for large-scale self-play
3. Inference speed for real-time Arena play
4. Model size constraints for deployment

## External Dependencies

### Game Engines
- **Forge**: Primary training environment - requires Java runtime
- **MTGJSON**: Card database - requires periodic updates
- **MTGA Client**: Deployment target - Windows/Mac specific

### ML Infrastructure
- GPU clusters for training (cloud-based)
- Docker for containerized Forge instances
- Distributed computing framework (Ray/Dask likely)

## Legal Considerations

- Ensure MTGA integration complies with Wizards of the Coast Terms of Service
- Forge is open-source but verify license compatibility
- MTGJSON data usage should follow their guidelines
- Consider intellectual property implications of novel deck discovery