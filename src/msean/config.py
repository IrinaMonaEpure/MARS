from pathlib import Path
import yaml


class Config(dict):
    """Stores sonfiguration parameters"""

    def __getattr__(self, key):
        """Ensures dot access (config.a.b instead of config['a']['b'])"""
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        
        # Nested dicts
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value

        return value
    

def load_config(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # TODO: validate parameter values (e.g. node/layer/affiliation num/square side positive,
    # layer num equal to affiliation num vector length, literals for distance, distributions,
    # required keys etc.)

    return Config(cfg)