"""Main CLI entry point for ManaMind."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(name="manamind", help="ManaMind - AI agent for Magic: The Gathering")
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console)]
    )


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to training config file"),
    forge_path: Optional[Path] = typer.Option(None, "--forge-path", help="Path to Forge installation"),
    iterations: int = typer.Option(100, "--iterations", "-i", help="Number of training iterations"),
    games_per_iteration: int = typer.Option(50, "--games", "-g", help="Games per training iteration"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), "--checkpoint-dir", help="Directory for checkpoints"),
    resume_from: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Train the ManaMind agent using self-play."""
    setup_logging(verbose)
    
    console.print("[bold blue]Starting ManaMind training...[/bold blue]")
    
    try:
        from manamind.training.self_play import SelfPlayTrainer
        from manamind.models.policy_value_network import create_policy_value_network
        from manamind.forge_interface import ForgeClient
        import torch
        
        # Create policy-value network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"Using device: {device}")
        
        network = create_policy_value_network().to(device)
        
        # Setup Forge client if path provided
        forge_client = None
        if forge_path:
            forge_client = ForgeClient(forge_path=forge_path)
        
        # Create trainer
        config = {
            'training_iterations': iterations,
            'games_per_iteration': games_per_iteration,
            'checkpoint_dir': str(checkpoint_dir),
        }
        
        trainer = SelfPlayTrainer(
            policy_value_network=network,
            forge_client=forge_client,
            config=config
        )
        
        # Resume from checkpoint if specified
        if resume_from:
            trainer.load_checkpoint(str(resume_from))
            console.print(f"Resumed from checkpoint: {resume_from}")
        
        # Start training
        trainer.train()
        
        console.print("[bold green]Training completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def eval(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    opponent: str = typer.Option("forge", "--opponent", "-o", help="Opponent type (forge, random, human)"),
    num_games: int = typer.Option(10, "--games", "-g", help="Number of evaluation games"),
    forge_path: Optional[Path] = typer.Option(None, "--forge-path", help="Path to Forge installation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Evaluate a trained ManaMind model."""
    setup_logging(verbose)
    
    console.print("[bold blue]Starting ManaMind evaluation...[/bold blue]")
    
    try:
        from manamind.evaluation.evaluator import ModelEvaluator
        import torch
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create evaluator
        evaluator = ModelEvaluator(
            model_checkpoint=checkpoint,
            forge_path=forge_path
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            opponent_type=opponent,
            num_games=num_games
        )
        
        # Print results
        console.print("[bold green]Evaluation Results:[/bold green]")
        console.print(f"Win Rate: {results['win_rate']:.1%}")
        console.print(f"Games Played: {results['total_games']}")
        console.print(f"Wins: {results['wins']}")
        console.print(f"Losses: {results['losses']}")
        console.print(f"Draws: {results['draws']}")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def forge_test(
    forge_path: Optional[Path] = typer.Option(None, "--forge-path", help="Path to Forge installation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Test connection to Forge game engine."""
    setup_logging(verbose)
    
    console.print("[bold blue]Testing Forge connection...[/bold blue]")
    
    try:
        from manamind.forge_interface import ForgeClient
        
        with ForgeClient(forge_path=forge_path) as client:
            console.print("[green]✓[/green] Successfully connected to Forge")
            
            # Try to create a test game
            game_id = client.create_game("deck1.dck", "deck2.dck")
            console.print(f"[green]✓[/green] Created test game: {game_id}")
            
            # Get initial game state
            state = client.get_game_state(game_id)
            console.print(f"[green]✓[/green] Retrieved game state")
            
            console.print("[bold green]Forge connection test successful![/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]Forge connection failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def play(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    deck_path: Optional[Path] = typer.Option(None, "--deck", help="Path to deck file"),
    opponent: str = typer.Option("human", "--opponent", "-o", help="Opponent type (human, forge, random)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Play a game against the ManaMind agent."""
    setup_logging(verbose)
    
    console.print("[bold blue]Starting ManaMind game...[/bold blue]")
    
    # TODO: Implement interactive play interface
    console.print("[yellow]Interactive play not yet implemented[/yellow]")


@app.command() 
def info():
    """Show ManaMind system information."""
    console.print("[bold blue]ManaMind System Information[/bold blue]")
    
    try:
        import torch
        from manamind import __version__
        
        console.print(f"Version: {__version__}")
        console.print(f"PyTorch Version: {torch.__version__}")
        console.print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            console.print(f"CUDA Device: {torch.cuda.get_device_name()}")
            console.print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check for optional dependencies
        try:
            import py4j
            console.print(f"Py4J Version: {py4j.__version__}")
        except ImportError:
            console.print("[yellow]Py4J not available (needed for Forge integration)[/yellow]")
        
        try:
            import jpype
            console.print(f"JPype Version: {jpype.__version__}")
        except ImportError:
            console.print("[yellow]JPype not available (alternative for Forge integration)[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error getting system info: {e}[/red]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()