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

def save_config(cfg, path):
    # TODO: Consider just copying file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert back to plain dict
    def to_dict(d):
        if isinstance(d, dict):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)