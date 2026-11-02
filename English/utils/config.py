import yaml


def load_configs(path):
    """Read the single config.yaml and return the (preprocess, model, train)
    triple that the FastSpeech 2 code expects."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["preprocess"], cfg["model"], cfg["train"]
