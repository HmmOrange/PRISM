import os
from pathlib import Path
import yaml
from pydantic import BaseModel


# =========================
# Load config.yaml safely
# =========================

ROOT_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_PATH / "configs" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)


# =========================
# DB settings model
# =========================

class DBSettings(BaseModel):
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


# =========================
# Public export
# Environment variables override config.yaml values
# =========================

db_settings = DBSettings(
    host=os.environ.get("DB_HOST", _config["db"]["host"]),
    port=int(os.environ.get("DB_PORT", _config["db"]["port"])),
    name=os.environ.get("DB_NAME", _config["db"]["name"]),
    user=os.environ.get("DB_USER", _config["db"]["user"]),
    password=os.environ.get("DB_PASSWORD", _config["db"]["password"]),
)
