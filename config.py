"""Configuration management for diraqBench"""
import os
from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Configuration for AI model providers"""
    model: str
    input_price: float  # Price per token
    output_price: float  # Price per token
    max_tokens: int
    temperature: float
    api_key_env: str


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    pdf_zoom: float = 2.0
    question_margin_above: int = 20
    question_margin_below: int = 10
    section_margin_above: int = 30
    section_margin_below: int = 10
    max_workers: int = 4  # For parallel processing
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class PathConfig:
    """Configuration for file paths"""
    data_dir: Path = Path("data")
    output_dir: Path = Path("extracted_questions")
    output_dir_2025: Path = Path("extracted_questions_2025")
    syllabus_file: Path = Path("data/jee_syllabus.json")
    answer_keys_dir: Path = Path("data/answer_keys")
    question_papers_dir: Path = Path("data/question_papers")
    
    def __post_init__(self):
        # Convert strings to Path objects if needed
        for field_name in ['data_dir', 'output_dir', 'output_dir_2025', 'syllabus_file', 'answer_keys_dir', 'question_papers_dir']:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))


class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use defaults"""
        default_config = self._get_default_config()
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                self._merge_config(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
                print("Using default configuration")
        
        self._create_configs(default_config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "models": {
                "anthropic": {
                    "claude-sonnet-4-20250514": {
                        "model": "claude-sonnet-4-20250514",
                        "input_price": 3/1000000,
                        "output_price": 15/1000000,
                        "max_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "ANTHROPIC_API_KEY"
                    },
                    "claude-opus-4-20250514": {
                        "model": "claude-opus-4-20250514",
                        "input_price": 15/1000000,
                        "output_price": 75/1000000,
                        "max_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "ANTHROPIC_API_KEY"
                    }
                },
                "openai": {
                    "gpt-4o": {
                        "model": "gpt-4o",
                        "input_price": 2.5/1000000,
                        "output_price": 10/1000000,
                        "max_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "OPENAI_API_KEY"
                    },
                    "gpt-4.1": {
                        "model": "gpt-4.1",
                        "input_price": 2/1000000,
                        "output_price": 8/1000000,
                        "max_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "OPENAI_API_KEY"
                    },
                    "o3": {
                        "model": "o3",
                        "input_price": 2/1000000, # o3 price drop by 80% on 11th June 2025
                        "output_price": 8/1000000, # o3 price drop by 80% on 11th June 2025
                        "max_tokens": 50000,
                        "temperature": 1.0,
                        "api_key_env": "OPENAI_API_KEY"
                    },
                    "o4-mini": {
                        "model": "o4-mini",
                        "input_price": 1.10/1000000,
                        "output_price": 4.40/1000000,
                        "max_tokens": 50000,
                        "temperature": 1.0,
                        "api_key_env": "OPENAI_API_KEY"
                    }
                },
                "google": {
                    "gemini-2.5-flash-preview-05-20": {
                        "model": "gemini-2.5-flash-preview-05-20",
                        "input_price": 0.15/1000000,
                        "output_price": 0.60/1000000,
                        "max_output_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "GOOGLE_API_KEY"
                    },
                    "gemini-2.5-pro-preview-06-05": {
                        "model": "gemini-2.5-pro-preview-06-05",
                        "input_price": 1.25/1000000,
                        "output_price": 10.00/1000000,
                        "max_output_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "GOOGLE_API_KEY"
                    },
                    "gemini-2.0-flash": {
                        "model": "gemini-2.0-flash",
                        "input_price": 0.10/1000000,
                        "output_price": 0.40/1000000,
                        "max_output_tokens": 50000,
                        "temperature": 0.0,
                        "api_key_env": "GOOGLE_API_KEY"
                    }
                }
            },
            "processing": {
                "pdf_zoom": 2.0,
                "question_margin_above": 20,
                "question_margin_below": 10,
                "section_margin_above": 30,
                "section_margin_below": 10,
                "max_workers": 4,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "data/outputs/extracted_questions",
                "output_dir_2025": "data/outputs/extracted_questions",
                "syllabus_file": "data/inputs/syllabus/jee_syllabus.json",
                "answer_keys_dir": "data/inputs/question_papers",
                "question_papers_dir": "data/inputs/question_papers"
            },
            "logging": {
                "level": "INFO",
                "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_file": "diraqbench.log",
                "console": True
            }
        }
    
    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]):
        """Recursively merge user config into default config"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _create_configs(self, config_dict: Dict[str, Any]):
        """Create configuration objects from dictionary"""
        # Model configurations - now supports multiple models per provider
        self.models = {}
        for provider, models_dict in config_dict["models"].items():
            self.models[provider] = {}
            for model_name, model_config in models_dict.items():
                # Handle different parameter names for different providers
                processed_config = model_config.copy()
                
                # Convert max_output_tokens to max_tokens for Google models
                if 'max_output_tokens' in processed_config and 'max_tokens' not in processed_config:
                    processed_config['max_tokens'] = processed_config.pop('max_output_tokens')
                
                self.models[provider][model_name] = ModelConfig(**processed_config)
        
        # Processing configuration
        self.processing = ProcessingConfig(**config_dict["processing"])
        
        # Path configuration
        self.paths = PathConfig(**config_dict["paths"])
        
        # Logging configuration
        self.logging = config_dict["logging"]
    
    def get_model_config(self, provider: str, model_name: str = None) -> ModelConfig:
        """Get model configuration for a provider and specific model"""
        if provider not in self.models:
            raise ValueError(f"Invalid provider: {provider}. Available: {list(self.models.keys())}")
        
        if model_name is None:
            # Return the first model if no specific model is requested (backward compatibility)
            model_name = list(self.models[provider].keys())[0]
        
        if model_name not in self.models[provider]:
            raise ValueError(f"Invalid model '{model_name}' for provider '{provider}'. Available: {list(self.models[provider].keys())}")
        
        return self.models[provider][model_name]
    
    def get_all_models_for_provider(self, provider: str) -> Dict[str, ModelConfig]:
        """Get all model configurations for a provider"""
        if provider not in self.models:
            raise ValueError(f"Invalid provider: {provider}. Available: {list(self.models.keys())}")
        return self.models[provider]
    
    def get_all_provider_model_combinations(self) -> List[tuple[str, str]]:
        """Get all (provider, model) combinations"""
        combinations = []
        for provider, models in self.models.items():
            for model_name in models.keys():
                combinations.append((provider, model_name))
        return combinations
    
    def validate_api_keys(self):
        """Validate that required API keys are set"""
        missing_keys = []
        api_keys_checked = set()
        
        for provider, models_dict in self.models.items():
            for model_name, model_config in models_dict.items():
                if model_config.api_key_env not in api_keys_checked:
                    if not os.getenv(model_config.api_key_env):
                        missing_keys.append(f"{provider}: {model_config.api_key_env}")
                    api_keys_checked.add(model_config.api_key_env)
        
        if missing_keys:
            raise EnvironmentError(f"Missing API keys: {', '.join(missing_keys)}")
    
    def save_config(self):
        """Save current configuration to file"""
        config_dict = {
            "models": {
                provider: {
                    model_name: {
                        "model": config.model,
                        "input_price": config.input_price,
                        "output_price": config.output_price,
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "api_key_env": config.api_key_env
                    }
                    for model_name, config in models_dict.items()
                }
                for provider, models_dict in self.models.items()
            },
            "processing": {
                "pdf_zoom": self.processing.pdf_zoom,
                "question_margin_above": self.processing.question_margin_above,
                "question_margin_below": self.processing.question_margin_below,
                "section_margin_above": self.processing.section_margin_above,
                "section_margin_below": self.processing.section_margin_below,
                "max_workers": self.processing.max_workers,
                "retry_attempts": self.processing.retry_attempts,
                "retry_delay": self.processing.retry_delay
            },
            "paths": {
                "data_dir": str(self.paths.data_dir),
                "output_dir": str(self.paths.output_dir),
                "output_dir_2025": str(self.paths.output_dir_2025),
                "syllabus_file": str(self.paths.syllabus_file),
                "answer_keys_dir": str(self.paths.answer_keys_dir),
                "question_papers_dir": str(self.paths.question_papers_dir)
            },
            "logging": self.logging
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


# Global configuration instance
config = Config()