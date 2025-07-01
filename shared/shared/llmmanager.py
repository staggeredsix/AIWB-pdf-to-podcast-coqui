"""Utility wrapper for running LLM inference with Ollama."""

from typing import List, Dict, Any, Optional, Union
import logging
import ujson as json
from shared.otel import OpenTelemetryInstrumentation
from opentelemetry.trace.status import StatusCode
from pathlib import Path
from dataclasses import dataclass
from langchain_core.messages import AIMessage
from ollama import Client, AsyncClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""

    name: str
    api_base: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing model configuration
            
        Returns:
            ModelConfig: New ModelConfig instance
        """
        return cls(
            name=data["name"],
            api_base=data["api_base"],
        )


class LLMManager:
    """
    Lightweight wrapper around the Ollama Python client.
    Provides a similar interface to the previous ChatNVIDIA-based manager while
    enabling local inference with Llama 3.1 8B Instruct.

    Configs can be overridden by providing a custom config file. Currently the defaults are
    hardcoded to build.nvidia.com endpoints.

    Attributes:
        api_key (str): API key for NVIDIA endpoints
        telemetry (OpenTelemetryInstrumentation): Telemetry instrumentation instance
        _clients (Dict[str, Client]): Cached synchronous clients per model
        _aclients (Dict[str, AsyncClient]): Cached async clients per model
        model_configs (Dict[str, ModelConfig]): Model configurations

    Usage:
    >>> llm_manager = LLMManager(api_key, telemetry)
    >>> llm_manager.query_sync("reasoning", [{"role": "user", "content": "Hello, world!"}], "test")
    """

    DEFAULT_CONFIGS = {
        "reasoning": {"name": "llama3:8b-instruct", "api_base": "http://ollama:11434"},
        "iteration": {"name": "llama3:8b-instruct", "api_base": "http://ollama:11434"},
        "json": {"name": "llama3:8b-instruct", "api_base": "http://ollama:11434"},
    }

    def __init__(
        self,
        api_key: str,
        telemetry: OpenTelemetryInstrumentation,
        config_path: Optional[str] = None,
    ):
        """
        Initialize LLMManager with telemetry.

        Args:
            api_key (str): API key for NVIDIA endpoints
            telemetry (OpenTelemetryInstrumentation): Telemetry instrumentation instance
            config_path (Optional[str]): Path to custom model configurations file

        Raises:
            Exception: If initialization fails
        """
        try:
            self.api_key = api_key
            self.telemetry = telemetry
            self._clients: Dict[str, Client] = {}
            self._aclients: Dict[str, AsyncClient] = {}
            self.model_configs = self._load_configurations(config_path)
            logger.info("Successfully initialized LLMManager")
        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    def _load_configurations(
        self, config_path: Optional[str]
    ) -> Dict[str, ModelConfig]:
        """Load model configurations from JSON file if provided, otherwise use defaults.
        
        Args:
            config_path (Optional[str]): Path to configuration JSON file
            
        Returns:
            Dict[str, ModelConfig]: Dictionary mapping model keys to configurations
        """
        configs = self.DEFAULT_CONFIGS.copy()
        if config_path:
            try:
                config_path = Path(config_path)
                if config_path.exists():
                    with config_path.open() as f:
                        custom_configs = json.load(f)
                    configs.update(custom_configs)
                else:
                    logger.warning(
                        f"Config file {config_path} not found, using default configurations"
                    )
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                logger.warning("Using default configurations")
        return {key: ModelConfig.from_dict(config) for key, config in configs.items()}

    def get_client(self, model_key: str) -> Client:
        """Get or create an Ollama client for the specified model key."""
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model key: {model_key}")
        if model_key not in self._clients:
            config = self.model_configs[model_key]
            self._clients[model_key] = Client(host=config.api_base)
        return self._clients[model_key]

    def get_async_client(self, model_key: str) -> AsyncClient:
        """Get or create an asynchronous Ollama client."""
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model key: {model_key}")
        if model_key not in self._aclients:
            config = self.model_configs[model_key]
            self._aclients[model_key] = AsyncClient(host=config.api_base)
        return self._aclients[model_key]

    def query_sync(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[AIMessage, Dict[str, Any]]:
        """Send a synchronous query to the specified model.
        
        Args:
            model_key (str): Key identifying which model to use
            messages (List[Dict[str, str]]): List of message dictionaries
            query_name (str): Name of query for telemetry
            json_schema (Optional[Dict]): Schema for structured output
            retries (int): Number of retry attempts
            
        Returns:
            Union[AIMessage, Dict[str, Any]]: Model response
            
        Raises:
            Exception: If query fails after retries
        """
        with self.telemetry.tracer.start_as_current_span(
            f"agent.query.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", False)

            try:
                client = self.get_client(model_key)
                config = self.model_configs[model_key]
                resp = client.chat(model=config.name, messages=messages)
                content = resp["message"]["content"]
                if json_schema:
                    return json.loads(content)
                return AIMessage(content=content)
            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Query failed: {e}")
                raise Exception(
                    f"Failed to get response after {retries} attempts"
                ) from e

    async def query_async(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[AIMessage, Dict[str, Any]]:
        """Send an asynchronous query to the specified model.
        
        Args:
            model_key (str): Key identifying which model to use
            messages (List[Dict[str, str]]): List of message dictionaries
            query_name (str): Name of query for telemetry
            json_schema (Optional[Dict]): Schema for structured output
            retries (int): Number of retry attempts
            
        Returns:
            Union[AIMessage, Dict[str, Any]]: Model response
            
        Raises:
            Exception: If query fails after retries
        """
        with self.telemetry.tracer.start_as_current_span(
            f"agent.query.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", True)

            try:
                client = self.get_async_client(model_key)
                config = self.model_configs[model_key]
                resp = await client.chat(model=config.name, messages=messages)
                content = resp["message"]["content"]
                if json_schema:
                    return json.loads(content)
                return AIMessage(content=content)
            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Query failed: {e}")
                raise Exception(
                    f"Failed to get response after {retries} attempts"
                ) from e

    def stream_sync(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[str, Dict[str, Any]]:
        """Send a synchronous streaming query to the specified model.
        
        Args:
            model_key (str): Key identifying which model to use
            messages (List[Dict[str, str]]): List of message dictionaries
            query_name (str): Name of query for telemetry
            json_schema (Optional[Dict]): Schema for structured output
            retries (int): Number of retry attempts
            
        Returns:
            Union[str, Dict[str, Any]]: Final chunk from model stream
            
        Raises:
            Exception: If streaming query fails after retries
        """
        with self.telemetry.tracer.start_as_current_span(
            f"agent.stream.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", False)

            try:
                client = self.get_client(model_key)
                config = self.model_configs[model_key]
                last_chunk = None
                for chunk in client.chat(model=config.name, messages=messages, stream=True):
                    last_chunk = chunk["message"]["content"]

                if json_schema:
                    return json.loads(last_chunk)
                return last_chunk

            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Streaming query failed: {e}")
                raise Exception(
                    f"Failed to get streaming response after {retries} attempts"
                ) from e

    async def stream_async(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        query_name: str,
        json_schema: Optional[Dict] = None,
        retries: int = 5,
    ) -> Union[str, Dict[str, Any]]:
        """Send an asynchronous streaming query to the specified model.
        
        Args:
            model_key (str): Key identifying which model to use
            messages (List[Dict[str, str]]): List of message dictionaries
            query_name (str): Name of query for telemetry
            json_schema (Optional[Dict]): Schema for structured output
            retries (int): Number of retry attempts
            
        Returns:
            Union[str, Dict[str, Any]]: Final chunk from model stream
            
        Raises:
            Exception: If streaming query fails after retries
        """
        with self.telemetry.tracer.start_as_current_span(
            f"agent.stream.{query_name}"
        ) as span:
            span.set_attribute("model_key", model_key)
            span.set_attribute("retries", retries)
            span.set_attribute("async", True)

            try:
                client = self.get_async_client(model_key)
                config = self.model_configs[model_key]

                last_chunk = None
                async for chunk in await client.chat(model=config.name, messages=messages, stream=True):
                    last_chunk = chunk["message"]["content"]

                if json_schema:
                    return json.loads(last_chunk)
                return last_chunk

            except Exception as e:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                logger.error(f"Async streaming query failed: {e}")
                raise Exception(
                    f"Failed to get streaming response after {retries} attempts"
                ) from e
