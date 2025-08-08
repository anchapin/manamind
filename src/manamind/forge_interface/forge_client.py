"""Forge game engine client for Python-Java communication.

This module handles the low-level communication with the Forge game engine
using Py4J or JPype for the Python-Java bridge.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from py4j.java_gateway import GatewayParameters, JavaGateway

    PY4J_AVAILABLE = True
except ImportError:
    PY4J_AVAILABLE = False
    JavaGateway = None

try:
    import jpype
    import jpype.imports

    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False
    jpype = None


logger = logging.getLogger(__name__)


class ForgeConnectionError(Exception):
    """Raised when connection to Forge fails."""

    pass


class ForgeClient:
    """Client for communicating with the Forge game engine.

    This class handles:
    1. Starting/stopping Forge instances
    2. Sending commands to Forge
    3. Receiving game state updates
    4. Managing multiple Forge processes for parallel training
    """

    def __init__(
        self,
        forge_path: Optional[Path] = None,
        java_opts: Optional[List[str]] = None,
        port: int = 25333,
        timeout: float = 30.0,
        use_py4j: bool = True,
    ):
        """Initialize Forge client.

        Args:
            forge_path: Path to Forge installation directory
            java_opts: Java options for running Forge
            port: Port for communication with Forge
            timeout: Connection timeout in seconds
            use_py4j: Whether to use Py4J (if False, uses JPype)
        """
        self.forge_path = forge_path or self._find_forge_installation()
        self.java_opts = java_opts or ["-Xmx4G", "-server"]
        self.port = port
        self.timeout = timeout
        self.use_py4j = use_py4j and PY4J_AVAILABLE

        # Runtime state
        self.forge_process: Optional[subprocess.Popen] = None
        self.gateway: Optional[Any] = None
        self.forge_api: Optional[Any] = None
        self.is_connected = False

        # Validate setup
        self._validate_setup()

    def _find_forge_installation(self) -> Path:
        """Try to find Forge installation automatically."""
        # Common Forge locations
        possible_paths = [
            Path("./forge"),
            Path("./forge-gui"),
            Path("/opt/forge"),
            Path.home() / "forge",
            Path.home() / "Downloads" / "forge-gui",
        ]

        for path in possible_paths:
            if path.exists() and (path / "forge-gui.jar").exists():
                logger.info(f"Found Forge installation at {path}")
                return path

        # If not found, return default path (user will need to install Forge)
        return Path("./forge")

    def _validate_setup(self) -> None:
        """Validate that required dependencies are available."""
        if not self.use_py4j and not JPYPE_AVAILABLE:
            msg = (
                "Neither Py4J nor JPype is available. Please install one:\n"
                "pip install py4j  # or\n"
                "pip install JPype1"
            )
            raise ForgeConnectionError(msg)

        if not self.forge_path.exists():
            logger.warning(
                f"Forge installation not found at {self.forge_path}. "
                "Please ensure Forge is installed and the path is correct."
            )

    def start_forge(self, headless: bool = True) -> None:
        """Start a Forge game engine instance.

        Args:
            headless: Whether to run Forge without GUI

        Raises:
            ForgeConnectionError: If Forge fails to start
        """
        if self.is_connected:
            logger.warning("Forge is already running")
            return

        logger.info(f"Starting Forge on port {self.port}")

        # Build command to start Forge
        forge_jar = self.forge_path / "forge-gui.jar"
        if not forge_jar.exists():
            raise ForgeConnectionError(f"Forge JAR not found: {forge_jar}")

        cmd = ["java"] + self.java_opts

        if headless:
            cmd.extend(["-Djava.awt.headless=true", "-Dforge.headless=true"])

        # Add ManaMind API mode
        cmd.extend(
            [
                "-Dforge.api.mode=true",
                f"-Dforge.api.port={self.port}",
                "-jar",
                str(forge_jar),
            ]
        )

        try:
            # Start Forge process
            self.forge_process = subprocess.Popen(
                cmd,
                cwd=self.forge_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for Forge to start
            self._wait_for_forge_startup()

            # Establish communication
            if self.use_py4j:
                self._connect_py4j()
            else:
                self._connect_jpype()

            logger.info("Forge started successfully")
            self.is_connected = True

        except Exception as e:
            self.stop_forge()
            raise ForgeConnectionError(f"Failed to start Forge: {e}")

    def _wait_for_forge_startup(self) -> None:
        """Wait for Forge to finish starting up."""
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            if self.forge_process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.forge_process.communicate()
                raise ForgeConnectionError(
                    f"Forge process terminated unexpectedly:\n"
                    f"STDOUT: {stdout}\nSTDERR: {stderr}"
                )

            # Check if Forge is ready (look for startup message in logs)
            # TODO: Implement proper startup detection
            time.sleep(1.0)

        if time.time() - start_time >= self.timeout:
            raise ForgeConnectionError("Forge startup timed out")

    def _connect_py4j(self) -> None:
        """Connect to Forge using Py4J."""
        if not PY4J_AVAILABLE:
            raise ForgeConnectionError("Py4J not available")

        try:
            gateway_params = GatewayParameters(
                port=self.port, auto_convert=True
            )
            self.gateway = JavaGateway(gateway_parameters=gateway_params)

            # Get the Forge API object
            self.forge_api = self.gateway.entry_point

            # Test connection
            version = self.forge_api.getVersion()
            logger.info(f"Connected to Forge version {version} via Py4J")

        except Exception as e:
            raise ForgeConnectionError(f"Failed to connect via Py4J: {e}")

    def _connect_jpype(self) -> None:
        """Connect to Forge using JPype."""
        if not JPYPE_AVAILABLE:
            raise ForgeConnectionError("JPype not available")

        try:
            # Start JVM if not already started
            if not jpype.isJVMStarted():
                jpype.startJVM(
                    jpype.getDefaultJVMPath(),
                    *self.java_opts,
                    classpath=[str(self.forge_path / "forge-gui.jar")],
                )

            # Import Forge API classes
            from forge.api import ManaMindAPI

            self.forge_api = ManaMindAPI()

            # Test connection
            version = self.forge_api.getVersion()
            logger.info(f"Connected to Forge version {version} via JPype")

        except Exception as e:
            raise ForgeConnectionError(f"Failed to connect via JPype: {e}")

    def stop_forge(self) -> None:
        """Stop the Forge game engine instance."""
        logger.info("Stopping Forge")

        # Close API connection
        if self.gateway:
            try:
                self.gateway.shutdown()
            except Exception:
                pass
            self.gateway = None

        if JPYPE_AVAILABLE and jpype.isJVMStarted():
            try:
                jpype.shutdownJVM()
            except Exception:
                pass

        # Terminate Forge process
        if self.forge_process:
            try:
                self.forge_process.terminate()
                self.forge_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Forge did not terminate gracefully, killing process"
                )
                self.forge_process.kill()
                self.forge_process.wait()
            except Exception as e:
                logger.error(f"Error stopping Forge process: {e}")

            self.forge_process = None

        self.forge_api = None
        self.is_connected = False
        logger.info("Forge stopped")

    def create_game(
        self,
        deck1_path: str,
        deck2_path: str,
        game_format: str = "Constructed",
    ) -> str:
        """Create a new game in Forge.

        Args:
            deck1_path: Path to player 1's deck file
            deck2_path: Path to player 2's deck file
            game_format: Game format (Constructed, Limited, etc.)

        Returns:
            Game ID for the created game

        Raises:
            ForgeConnectionError: If game creation fails
        """
        if not self.is_connected:
            raise ForgeConnectionError("Not connected to Forge")

        try:
            game_id = self.forge_api.createGame(
                deck1_path, deck2_path, game_format
            )
            logger.info(f"Created game {game_id}")
            return game_id

        except Exception as e:
            raise ForgeConnectionError(f"Failed to create game: {e}")

    def get_game_state(self, game_id: str) -> Dict[str, Any]:
        """Get the current game state from Forge.

        Args:
            game_id: ID of the game

        Returns:
            Game state as a dictionary

        Raises:
            ForgeConnectionError: If getting state fails
        """
        if not self.is_connected:
            raise ForgeConnectionError("Not connected to Forge")

        try:
            state_json = self.forge_api.getGameState(game_id)
            return json.loads(state_json)

        except Exception as e:
            raise ForgeConnectionError(f"Failed to get game state: {e}")

    def send_action(self, game_id: str, action_data: Dict[str, Any]) -> bool:
        """Send an action to Forge.

        Args:
            game_id: ID of the game
            action_data: Action data as dictionary

        Returns:
            True if action was accepted

        Raises:
            ForgeConnectionError: If sending action fails
        """
        if not self.is_connected:
            raise ForgeConnectionError("Not connected to Forge")

        try:
            action_json = json.dumps(action_data)
            result = self.forge_api.sendAction(game_id, action_json)
            return result

        except Exception as e:
            raise ForgeConnectionError(f"Failed to send action: {e}")

    def get_legal_actions(self, game_id: str) -> List[Dict[str, Any]]:
        """Get legal actions for the current player.

        Args:
            game_id: ID of the game

        Returns:
            List of legal actions as dictionaries

        Raises:
            ForgeConnectionError: If getting actions fails
        """
        if not self.is_connected:
            raise ForgeConnectionError("Not connected to Forge")

        try:
            actions_json = self.forge_api.getLegalActions(game_id)
            return json.loads(actions_json)

        except Exception as e:
            raise ForgeConnectionError(f"Failed to get legal actions: {e}")

    def is_game_over(self, game_id: str) -> bool:
        """Check if a game has ended.

        Args:
            game_id: ID of the game

        Returns:
            True if game is over
        """
        if not self.is_connected:
            return True

        try:
            return self.forge_api.isGameOver(game_id)
        except Exception as e:
            logger.error(f"Error checking if game is over: {e}")
            return True

    def get_winner(self, game_id: str) -> Optional[int]:
        """Get the winner of a finished game.

        Args:
            game_id: ID of the game

        Returns:
            Winner player ID (0 or 1), or None if draw/ongoing
        """
        if not self.is_connected:
            return None

        try:
            return self.forge_api.getWinner(game_id)
        except Exception as e:
            logger.error(f"Error getting winner: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start_forge()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_forge()
