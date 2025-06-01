from abc import ABC, abstractmethod


class LearningAlgorithm(ABC):
    """
    Abstract base class for learning algorithms in potential games.

    Subclasses must implement the run method.
    """

    @classmethod
    @abstractmethod
    def run(cls, game: "GameEngine", *args, **kwargs) -> None:
        """
        Run the learning algorithm on the provided game.

        Args:
            game (GameEngine): The game engine instance to run the algorithm on.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass