"""Configuration loader for FinShield."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel
from functools import lru_cache


class Config:
    """Global configuration singleton."""
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.getenv("FINSHIELD_CONFIG", "config.yaml")

        path = Path(config_path)
        if not path.is_absolute():
            # Look for config in project root
            project_root = Path(__file__).parent.parent.parent
            path = project_root / config_path

        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Substitute environment variables
        self._substitute_env_vars(self._config)

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR} with environment variables."""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key (e.g., 'ingestion.api.port')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def __getattr__(self, name: str) -> Any:
        return self.get(name)


@lru_cache()
def get_config() -> Config:
    """Get global config instance."""
    return Config()


# Pydantic models for validated config sections
class IngestionConfig(BaseModel):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    duckdb_path: str = "data/finshield.duckdb"
    kafka_servers: str = "localhost:9092"


class DetectionConfig(BaseModel):
    pyod_models: list = []
    xgboost_params: dict = {}
    redis_host: str = "localhost"
    redis_port: int = 6379


class GraphConfig(BaseModel):
    min_cycle_length: int = 3
    max_cycle_length: int = 10
    kingpin_min_in_degree: int = 5


class RiskConfig(BaseModel):
    ml_weight: float = 0.4
    graph_weight: float = 0.35
    regulatory_weight: float = 0.25
    watch_threshold: int = 30
    alert_threshold: int = 60
    act_threshold: int = 85


class AgentConfig(BaseModel):
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    chromadb_path: str = "data/chromadb"


class InterfaceConfig(BaseModel):
    streamlit_host: str = "0.0.0.0"
    streamlit_port: int = 8501


def get_ingestion_config() -> IngestionConfig:
    cfg = get_config()
    return IngestionConfig(
        api_host=cfg.get("ingestion.api.host", "0.0.0.0"),
        api_port=cfg.get("ingestion.api.port", 8000),
        duckdb_path=cfg.get("ingestion.duckdb.path", "data/finshield.duckdb"),
        kafka_servers=cfg.get("ingestion.kafka.bootstrap_servers", "localhost:9092"),
    )


def get_detection_config() -> DetectionConfig:
    cfg = get_config()
    return DetectionConfig(
        pyod_models=cfg.get("detection.pyod.models", []),
        xgboost_params=cfg.get("detection.xgboost", {}),
        redis_host=cfg.get("detection.redis.host", "localhost"),
        redis_port=cfg.get("detection.redis.port", 6379),
    )


def get_graph_config() -> GraphConfig:
    cfg = get_config()
    return GraphConfig(
        min_cycle_length=cfg.get("graph.networkx.min_cycle_length", 3),
        max_cycle_length=cfg.get("graph.networkx.max_cycle_length", 10),
        kingpin_min_in_degree=cfg.get("graph.kingpin.min_in_degree", 5),
    )


def get_risk_config() -> RiskConfig:
    cfg = get_config()
    weights = cfg.get("risk.weights", {})
    thresholds = cfg.get("risk.thresholds", {})
    return RiskConfig(
        ml_weight=weights.get("ml_score", 0.4),
        graph_weight=weights.get("graph_score", 0.35),
        regulatory_weight=weights.get("regulatory_score", 0.25),
        watch_threshold=thresholds.get("watch", 30),
        alert_threshold=thresholds.get("alert", 60),
        act_threshold=thresholds.get("act", 85),
    )


def get_agent_config() -> AgentConfig:
    cfg = get_config()
    return AgentConfig(
        ollama_host=cfg.get("agent.ollama.host", "http://localhost:11434"),
        ollama_model=cfg.get("agent.ollama.model", "llama3:8b"),
        chromadb_path=cfg.get("agent.rag.chromadb_path", "data/chromadb"),
    )


def get_interface_config() -> InterfaceConfig:
    cfg = get_config()
    return InterfaceConfig(
        streamlit_host=cfg.get("interface.streamlit.host", "0.0.0.0"),
        streamlit_port=cfg.get("interface.streamlit.port", 8501),
    )