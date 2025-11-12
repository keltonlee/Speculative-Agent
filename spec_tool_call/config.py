"""Configuration for speculative tool calling framework."""
import os
from dataclasses import dataclass


@dataclass
class SpecConfig:
    """Configuration for speculative execution."""
    # Execution mode
    enable_speculation: bool = True  # Set to False for baseline actor-only mode
    
    # Models
    actor_model: str = "gpt-5"
    spec_model: str = "gpt-5-mini"
    model_provider: str = "openai"  # openai, google-genai, etc.

    # Speculation parameters
    top_k_spec: int = 3
    conf_threshold: float = 0.35

    # Execution limits
    max_steps: int = 12

    # LLM parameters
    llm_max_tokens: int = 1024
    
    # API Keys
    gensee_api_key: str = ""  # Gensee AI API key for web search

    @classmethod
    def from_env(cls) -> "SpecConfig":
        """Load configuration from environment variables."""
        return cls(
            enable_speculation=(os.getenv("DISABLE_SPECULATION", "0") != "1"),
            actor_model=os.getenv("GAIA_ACTOR_MODEL", "gpt-5"),
            spec_model=os.getenv("GAIA_SPEC_MODEL", "gpt-5-mini"),
            model_provider=os.getenv("GAIA_MODEL_PROVIDER", "openai"),
            top_k_spec=int(os.getenv("GAIA_TOPK", "3")),
            conf_threshold=float(os.getenv("GAIA_CONF_TH", "0.35")),
            max_steps=int(os.getenv("GAIA_MAX_STEPS", "12")),
            gensee_api_key=os.getenv("GENSEE_API_KEY", ""),
        )


# Global config instance
config = SpecConfig.from_env()
