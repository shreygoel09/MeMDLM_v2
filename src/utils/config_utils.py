from pathlib import Path
from omegaconf import OmegaConf

# Repository root (MeMDLM_v2/)
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_name: str):
    """Load a YAML config from src/configs/."""
    return OmegaConf.load(REPO_ROOT / "src" / "configs" / config_name)


def repo_path(*parts: str) -> Path:
    """Build an absolute path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)
