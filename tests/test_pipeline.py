"""
Basic tests for the Interaction Transcript Summarization pipeline.
"""
import pytest
from pathlib import Path


def test_project_structure():
    """Test that essential directories exist."""
    assert Path("src/interaction_transcript_summarization").exists()
    assert Path("config/config.yaml").exists()
    assert Path("params.yaml").exists()
    assert Path("main.py").exists()
    assert Path("app.py").exists()


def test_config_files():
    """Test that config files are valid YAML."""
    import yaml
    
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        assert "artifacts_root" in config
        assert "data_ingestion" in config
    
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        assert "TrainingArguments" in params


def test_imports():
    """Test that essential modules can be imported."""
    from src.interaction_transcript_summarization.logging import logger
    from src.interaction_transcript_summarization.entity import (
        DataIngestionConfig,
        DataValidationConfig,
        DataTransformationConfig
    )
    assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
