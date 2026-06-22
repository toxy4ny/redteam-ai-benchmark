"""YAML configuration management for benchmark."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_QUESTIONS_FILE = "datasets/v2/benchmark.jsonl"
DEFAULT_ANSWERS_FILE = "answers_all.txt"
DEFAULT_SCORER = "rubric"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    endpoint: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    default_model: Optional[str] = None
    timeout: int = 120
    keep_alive: Optional[str] = None


@dataclass
class ScoringConfig:
    """Configuration for scoring system."""

    method: str = DEFAULT_SCORER


@dataclass
class ExportConfig:
    """Configuration for export system."""

    formats: List[str] = field(default_factory=lambda: ["json"])
    output_dir: str = "."
    include_response: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization."""

    enabled: bool = False
    optimizer_model: Optional[str] = None
    optimizer_endpoint: Optional[str] = None
    max_iterations: int = 3
    strategies: List[str] = field(default_factory=lambda: [
        "role_playing",
        "technical",
        "few_shot",
        "cve_framing",
    ])


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability."""

    enabled: bool = False
    secret_key: Optional[str] = None
    public_key: Optional[str] = None
    host: str = "http://localhost:3000"


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    provider: ProviderConfig
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    questions_file: str = DEFAULT_QUESTIONS_FILE
    answers_file: str = DEFAULT_ANSWERS_FILE
    rate_limit_delay: float = 1.5
    max_tokens: int = 768
    temperature: float = 0.2
    concurrency: int = 1
    request_log: Optional[str] = None


# Default providers configuration
DEFAULT_PROVIDERS = {
    "lmstudio": ProviderConfig(
        name="lmstudio",
        endpoint="http://localhost:1234",
    ),
    "ollama": ProviderConfig(
        name="ollama",
        endpoint="http://localhost:11434",
    ),
    "openwebui": ProviderConfig(
        name="openwebui",
        endpoint="http://localhost:3000",
        api_key_env="OPENWEBUI_API_KEY",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        endpoint="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="anthropic/claude-4.5-haiku",
    ),
}


def _dict_to_provider_config(data: Dict[str, Any]) -> ProviderConfig:
    """Convert dict to ProviderConfig."""
    return ProviderConfig(
        name=data.get("name", "unknown"),
        endpoint=data.get("endpoint", ""),
        api_key=data.get("api_key"),
        api_key_env=data.get("api_key_env"),
        default_model=data.get("default_model"),
        timeout=data.get("timeout", 120),
        keep_alive=data.get("keep_alive"),
    )


def _dict_to_scoring_config(data: Dict[str, Any]) -> ScoringConfig:
    """Convert dict to ScoringConfig."""
    return ScoringConfig(
        method=data.get("method", DEFAULT_SCORER),
    )


def _dict_to_export_config(data: Dict[str, Any]) -> ExportConfig:
    """Convert dict to ExportConfig."""
    return ExportConfig(
        formats=[fmt.lower() for fmt in data.get("formats", ["json"])],
        output_dir=data.get("output_dir", "."),
        include_response=data.get("include_response", True),
    )


def _dict_to_optimization_config(data: Dict[str, Any]) -> OptimizationConfig:
    """Convert dict to OptimizationConfig."""
    return OptimizationConfig(
        enabled=data.get("enabled", False),
        optimizer_model=data.get("optimizer_model"),
        optimizer_endpoint=data.get("optimizer_endpoint"),
        max_iterations=data.get("max_iterations", 3),
        strategies=data.get("strategies", [
            "role_playing", "technical", "few_shot", "cve_framing"
        ]),
    )


def _dict_to_langfuse_config(data: Dict[str, Any]) -> LangfuseConfig:
    """Convert dict to LangfuseConfig. Auto-enables if keys are present."""
    secret_key = data.get("secret_key")
    public_key = data.get("public_key")
    # Auto-enable if both keys are present
    auto_enabled = bool(secret_key and public_key)
    return LangfuseConfig(
        enabled=data.get("enabled", auto_enabled),
        secret_key=secret_key,
        public_key=public_key,
        host=data.get("host", "http://localhost:3000"),
    )


def load_config(config_path: str) -> BenchmarkConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        BenchmarkConfig instance
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty config file: {config_path}")

    # Parse provider config
    provider_data = data.get("provider", {})
    provider_name = provider_data.get("name", "ollama")

    # Start with defaults if available
    if provider_name in DEFAULT_PROVIDERS:
        default_provider = DEFAULT_PROVIDERS[provider_name]
        provider = ProviderConfig(
            name=provider_name,
            endpoint=provider_data.get("endpoint", default_provider.endpoint),
            api_key=provider_data.get("api_key", default_provider.api_key),
            api_key_env=provider_data.get("api_key_env", default_provider.api_key_env),
            default_model=provider_data.get(
                "default_model", default_provider.default_model
            ),
            timeout=provider_data.get("timeout", default_provider.timeout),
            keep_alive=provider_data.get("keep_alive", default_provider.keep_alive),
        )
    else:
        provider = _dict_to_provider_config(provider_data)

    # Parse other configs
    scoring = _dict_to_scoring_config(data.get("scoring", {}))
    export = _dict_to_export_config(data.get("export", {}))
    optimization = _dict_to_optimization_config(data.get("optimization", {}))
    langfuse = _dict_to_langfuse_config(data.get("langfuse", {}))

    config = BenchmarkConfig(
        provider=provider,
        scoring=scoring,
        export=export,
        optimization=optimization,
        langfuse=langfuse,
        questions_file=data.get("questions_file", DEFAULT_QUESTIONS_FILE),
        answers_file=data.get("answers_file", DEFAULT_ANSWERS_FILE),
        rate_limit_delay=data.get("rate_limit_delay", 1.5),
        max_tokens=data.get("max_tokens", 768),
        temperature=data.get("temperature", 0.2),
        concurrency=data.get("concurrency", 1),
        request_log=data.get("request_log"),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    """Validate config values that affect runtime behavior."""
    if config.rate_limit_delay < 0:
        raise ValueError("rate_limit_delay must be >= 0")
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if config.temperature < 0:
        raise ValueError("temperature must be >= 0")
    if config.concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if config.provider.timeout <= 0:
        raise ValueError("provider.timeout must be > 0")

    if config.scoring.method != DEFAULT_SCORER:
        raise ValueError(f"Unsupported scoring method: {config.scoring.method}")

    unsupported_formats = set(config.export.formats) - {"json", "csv", "criteria_csv"}
    if unsupported_formats:
        formatted = ", ".join(sorted(unsupported_formats))
        raise ValueError(f"Unsupported export format(s): {formatted}")


def create_default_config(
    provider: str = "ollama",
    model: Optional[str] = None,
) -> BenchmarkConfig:
    """
    Create a default configuration.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        BenchmarkConfig with defaults
    """
    provider_config = DEFAULT_PROVIDERS.get(
        provider,
        ProviderConfig(name=provider, endpoint="http://localhost:11434"),
    )

    if model:
        provider_config = ProviderConfig(
            name=provider_config.name,
            endpoint=provider_config.endpoint,
            api_key_env=provider_config.api_key_env,
            default_model=model,
            timeout=provider_config.timeout,
            keep_alive=provider_config.keep_alive,
        )

    return BenchmarkConfig(provider=provider_config)


def save_config(config: BenchmarkConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: BenchmarkConfig instance
        config_path: Path to save YAML file
    """
    data = {
        "provider": {
            "name": config.provider.name,
            "endpoint": config.provider.endpoint,
        },
        "scoring": {
            "method": config.scoring.method,
        },
        "export": {
            "formats": config.export.formats,
            "output_dir": config.export.output_dir,
        },
        "optimization": {
            "enabled": config.optimization.enabled,
            "max_iterations": config.optimization.max_iterations,
        },
        "questions_file": config.questions_file,
        "answers_file": config.answers_file,
        "rate_limit_delay": config.rate_limit_delay,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "concurrency": config.concurrency,
        "request_log": config.request_log,
    }

    # Add optional fields
    if config.provider.api_key_env:
        data["provider"]["api_key_env"] = config.provider.api_key_env
    if config.provider.default_model:
        data["provider"]["default_model"] = config.provider.default_model
    if config.provider.keep_alive:
        data["provider"]["keep_alive"] = config.provider.keep_alive
    if config.optimization.optimizer_model:
        data["optimization"]["optimizer_model"] = config.optimization.optimizer_model

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_api_key(provider_config: ProviderConfig) -> Optional[str]:
    """
    Get API key from environment variable.

    Args:
        provider_config: Provider configuration

    Returns:
        API key or None
    """
    if provider_config.api_key_env:
        return os.environ.get(provider_config.api_key_env)
    return None
