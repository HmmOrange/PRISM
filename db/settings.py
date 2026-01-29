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
# =========================

db_settings = DBSettings(
    host=_config["db"]["host"],
    port=_config["db"]["port"],
    name=_config["db"]["name"],
    user=_config["db"]["user"],
    password=_config["db"]["password"],
)
