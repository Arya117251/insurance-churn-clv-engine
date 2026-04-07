from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    @abstractmethod
    def generate_brief(self, prompt: str) -> str:
        """
        Generate a retention brief based on the provided prompt.

        Args:
            prompt: The input prompt for the LLM

        Returns:
            Generated text response from the LLM
        """
        pass
