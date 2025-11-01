from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from appdirs import user_cache_dir

from mlebench.grade_helpers import Grader
from mlebench.utils import (
    get_logger,
    get_module_dir,
    get_repo_dir,
    import_fn,
    load_yaml,
)

logger = get_logger(__name__)


DEFAULT_DATA_DIR = (Path(user_cache_dir()) / "mle-bench" / "data").resolve()


@dataclass(frozen=True)
class Competition:
    id: str
    name: str
    description: str
    grader: Grader
    lite_grader: Grader
    answers: Path
    lite_answers: Path
    gold_submission: Path
    sample_submission: Path
    competition_type: str
    prepare_fn: Callable[[Path, Path, Path], Path]
    lite_prepare_fn: Callable[[Path, Path, Path], Path]
    raw_dir: Path
    private_dir: Path
    public_dir: Path
    private_lite_dir: Path
    public_lite_dir: Path
    checksums: Path
    leaderboard: Path

    def __post_init__(self):
        assert isinstance(self.id, str), "Competition id must be a string."
        assert isinstance(self.name, str), "Competition name must be a string."
        assert isinstance(
            self.description, str
        ), "Competition description must be a string."
        assert isinstance(
            self.grader, Grader
        ), "Competition grader must be of type Grader."
        assert isinstance(self.answers, Path), "Competition answers must be a Path."
        assert isinstance(self.gold_submission, Path), "Gold submission must be a Path."
        assert isinstance(
            self.sample_submission, Path
        ), "Sample submission must be a Path."
        assert isinstance(
            self.competition_type, str
        ), "Competition type must be a string."
        assert isinstance(self.checksums, Path), "Checksums must be a Path."
        assert isinstance(self.leaderboard, Path), "Leaderboard must be a Path."
        assert len(self.id) > 0, "Competition id cannot be empty."
        assert len(self.name) > 0, "Competition name cannot be empty."
        assert len(self.description) > 0, "Competition description cannot be empty."
        assert len(self.competition_type) > 0, "Competition type cannot be empty."

    @staticmethod
    def from_dict(data: dict) -> "Competition":
        grader = Grader.from_dict(data["grader"])
        lite_grader = Grader.from_dict(data["lite_grader"])

        try:
            return Competition(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                grader=grader,
                lite_grader=lite_grader,
                answers=data["answers"],
                lite_answers=data["lite_answers"],
                sample_submission=data["sample_submission"],
                gold_submission=data["gold_submission"],
                competition_type=data["competition_type"],
                prepare_fn=data["prepare_fn"],
                lite_prepare_fn=data["lite_prepare_fn"],
                raw_dir=data["raw_dir"],
                public_dir=data["public_dir"],
                private_dir=data["private_dir"],
                public_lite_dir=data["public_lite_dir"],
                private_lite_dir=data["private_lite_dir"],
                checksums=data["checksums"],
                leaderboard=data["leaderboard"],
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in competition config!")


class Registry:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._data_dir = data_dir.resolve()

    def get_competition(self, competition_id: str) -> Competition:
        """Fetch the competition from the registry."""

        # Handle lite competitions
        is_lite = competition_id.endswith("_lite")
        base_competition_id = (
            competition_id.replace("_lite", "") if is_lite else competition_id
        )

        config_path = self.get_competitions_dir() / base_competition_id / "config.yaml"
        config = load_yaml(config_path)

        checksums_path = (
            self.get_competitions_dir() / base_competition_id / "checksums.yaml"
        )
        leaderboard_path = (
            self.get_competitions_dir() / base_competition_id / "leaderboard.csv"
        )

        description_path = get_repo_dir() / config["description"]
        description = description_path.read_text(encoding="utf-8")

        preparer_fn = import_fn(config["preparer"])
        lite_preparer_fn = import_fn(config["lite_preparer"])

        # Use lite paths if this is a lite competition
        data_dir = (
            self.get_data_dir() / competition_id
            if is_lite
            else self.get_data_dir() / base_competition_id
        )

        answers = data_dir / "prepared" / "private" / "answers.csv"
        lite_answers = data_dir / "prepared_lite" / "private" / "answers.csv"
        gold_submission = answers
        if "gold_submission" in config["dataset"]:
            # For lite competitions, adjust gold submission path
            if is_lite:
                gold_submission = data_dir / "prepared" / "private" / "answers.csv"
            else:
                gold_submission = (
                    self.get_data_dir() / config["dataset"]["gold_submission"]
                )

        sample_submission = data_dir / "prepared" / "public" / "sample_submission.csv"

        raw_dir = data_dir / "raw"
        private_dir = data_dir / "prepared" / "private"
        public_dir = data_dir / "prepared" / "public"
        
        private_lite_dir = data_dir / "prepared_lite" / "private"
        public_lite_dir = data_dir / "prepared_lite" / "public"

        return Competition.from_dict(
            {
                **config,
                "id": competition_id,  # Use the full ID including _lite suffix
                "description": description,
                "answers": answers,
                "lite_answers": lite_answers,
                "sample_submission": sample_submission,
                "gold_submission": gold_submission,
                "prepare_fn": preparer_fn,
                "lite_prepare_fn": lite_preparer_fn,
                "raw_dir": raw_dir,
                "private_dir": private_dir,
                "public_dir": public_dir,
                "private_lite_dir": private_lite_dir,
                "public_lite_dir": public_lite_dir,
                "checksums": checksums_path,
                "leaderboard": leaderboard_path,
            }
        )

    def get_competitions_dir(self) -> Path:
        """Retrieves the competition directory within the registry."""

        return get_module_dir() / "competitions"

    def get_splits_dir(self) -> Path:
        """Retrieves the splits directory within the repository."""

        return get_repo_dir() / "experiments" / "splits"

    def get_lite_competition_ids(self) -> list[str]:
        """List all competition IDs for the lite version (low complexity competitions)."""

        lite_competitions_file = self.get_splits_dir() / "low.txt"
        with open(lite_competitions_file, "r") as f:
            competition_ids = f.read().splitlines()
        return competition_ids

    def get_data_dir(self) -> Path:
        """Retrieves the data directory within the registry."""

        return self._data_dir

    def set_data_dir(self, new_data_dir: Path) -> "Registry":
        """Sets the data directory within the registry."""

        return Registry(new_data_dir)

    def list_competition_ids(self) -> list[str]:
        """List all competition IDs available in the registry, including lite versions, sorted alphabetically."""

        competition_configs = self.get_competitions_dir().rglob("config.yaml")
        base_competition_ids = [f.parent.stem for f in sorted(competition_configs)]

        # Add lite competition IDs if they exist
        all_competition_ids = base_competition_ids.copy()

        for comp_id in base_competition_ids:
            lite_dir = self.get_data_dir() / f"{comp_id}_lite"
            if lite_dir.exists() and (lite_dir / "prepared").exists():
                all_competition_ids.append(f"{comp_id}_lite")

        return sorted(all_competition_ids)


registry = Registry()
