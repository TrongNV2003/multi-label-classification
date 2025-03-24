from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

class LLMConfig(BaseSettings):
    base_url: str = Field(
        description="Base URL for OpenAI API",
        alias="LLM_URL",
    )
    api_key: str = Field(
        description="API key for OpenAI",
        alias="LLM_KEY",
    )
    model: str = Field(
        description="Model name to be used (e.g., GPT-4)",
        alias="LLM_MODEL",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature; higher values make output more random",
        alias="TEMPERATURE",
    )
    max_tokens: int = Field(
        default=1024,
        alias="MAX_TOKENS",
        description="Maximum number of tokens for API responses",
    )
    top_p: float = Field(
        default=0.95,
        alias="TOP_P",
        description="Nucleus sampling parameter; higher values increase randomness",
    )
    seed: int = Field(default=42, alias="SEED", description="Random seed for sampling")

llm_config = LLMConfig()
