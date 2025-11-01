import click
import json
from pathlib import Path
import asyncio
import shutil
import random
import pandas as pd

from mlebench.data import download_and_prepare_dataset, ensure_leaderboard_exists, prepare_lite_dataset
from mlebench.grade import grade_csv, grade_jsonl
from mlebench.registry import registry
from mlebench.utils import get_logger, validate_save_dir, read_csv
from mlebench.operations import generate_and_run, run_and_submit, submit_results

logger = get_logger(__name__)


@click.group()
def cli():
    """Runs agents on Kaggle competitions."""


@cli.command("prepare")
@click.option(
    "-c",
    "--competition-id",
    help=f"ID of the competition to prepare. Valid options include: {', '.join(registry.list_competition_ids()[:5])}... (and more)",
    type=str,
    required=False,
)
@click.option(
    "-a",
    "--all",
    is_flag=True,
    help="Prepare all competitions.",
)
@click.option(
    "--lite",
    is_flag=True,
    help="Prepare all the low complexity competitions (MLE-bench Lite).",
)
@click.option(
    "-l",
    "--list",
    type=click.Path(exists=True),
    help="Prepare a list of competitions specified line by line in a text file.",
)
@click.option(
    "--keep-raw",
    is_flag=True,
    help="Keep the raw competition files after the competition has been prepared.",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the data will be stored.",
)
@click.option(
    "--overwrite-checksums",
    is_flag=True,
    help="[For Developers] Overwrite the checksums file for the competition.",
)
@click.option(
    "--overwrite-leaderboard",
    is_flag=True,
    help="[For Developers] Overwrite the leaderboard file for the competition.",
)
@click.option(
    "--skip-verification",
    is_flag=True,
    help="[For Developers] Skip the verification of the checksums.",
)
def prepare(
    competition_id,
    all,
    lite,
    list,
    keep_raw,
    data_dir,
    overwrite_checksums,
    overwrite_leaderboard,
    skip_verification,
):
    """Download and prepare competitions for the MLE-bench dataset."""
    new_registry = registry.set_data_dir(Path(data_dir))

    if lite:
        competitions = [
            new_registry.get_competition(comp_id)
            for comp_id in new_registry.get_lite_competition_ids()
        ]
    elif all:
        competitions = [
            new_registry.get_competition(comp_id)
            for comp_id in registry.list_competition_ids()
        ]
    elif list:
        with open(list, "r") as f:
            competition_ids = f.read().splitlines()
        competitions = [
            new_registry.get_competition(comp_id) for comp_id in competition_ids
        ]
    else:
        if not competition_id:
            raise click.BadParameter(
                "One of --lite, --all, --list, or --competition-id must be specified."
            )
        competitions = [new_registry.get_competition(competition_id)]

    for competition in competitions:
        download_and_prepare_dataset(
            competition=competition,
            keep_raw=keep_raw,
            overwrite_checksums=overwrite_checksums,
            overwrite_leaderboard=overwrite_leaderboard,
            skip_verification=skip_verification,
        )

@cli.command("prepare_lite")
@click.option(
    "-c",
    "--competition-id",
    help=f"ID of the competition to prepare in lite version. Valid options include: {', '.join(registry.list_competition_ids()[:5])}... (and more)",
    type=str,
    required=False,
)
@click.option(
    "-a",
    "--all",
    is_flag=True,
    help="Prepare all competitions in lite version.",
)
@click.option(
    "--lite",
    is_flag=True,
    help="Prepare all the low complexity competitions in lite version (MLE-bench Lite).",
)
@click.option(
    "-l",
    "--list",
    type=click.Path(exists=True),
    help="Prepare a list of competitions specified line by line in a text file in lite version.",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the original data is stored and lite data will be created.",
)
@click.option(
    "--max-samples",
    type=int,
    default=100,
    help="Maximum number of samples to include in the lite dataset (default: 1000).",
)
def prepare_lite(competition_id, all, lite, list, data_dir, max_samples):
    """Create lite versions of prepared competitions with limited samples."""
    new_registry = registry.set_data_dir(Path(data_dir))

    if lite:
        competitions = [
            new_registry.get_competition(comp_id)
            for comp_id in new_registry.get_lite_competition_ids()
        ]
    elif all:
        competitions = [
            new_registry.get_competition(comp_id)
            for comp_id in registry.list_competition_ids()
        ]
    elif list:
        with open(list, "r") as f:
            competition_ids = f.read().splitlines()
        competitions = [
            new_registry.get_competition(comp_id) for comp_id in competition_ids
        ]
    else:
        if not competition_id:
            raise click.BadParameter(
                "One of --lite, --all, --list, or --competition-id must be specified."
            )
        competitions = [new_registry.get_competition(competition_id)]

    for competition in competitions:
        if competition.id == "denoising-dirty-documents":
            continue
        prepare_lite_dataset(
            competition=competition,
            max_test_samples=max_samples,
        )


@cli.command("grade")
@click.option(
    "--submission",
    type=click.Path(exists=True),
    required=True,
    help="Path to the JSONL file of submissions. Refer to README.md#submission-format for the required format.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Path to the directory where the evaluation metrics will be saved.",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the data used for grading is stored.",
)
@click.option(
    "--normalize-metrics",
    is_flag=True,
    help="Normalize metrics to the range [0, 1] before saving results.",
)
def grade(submission, output_dir, data_dir, normalize_metrics):
    """Grade a submission to the eval, comprising of several competition submissions."""
    new_registry = registry.set_data_dir(Path(data_dir))
    submission_path = Path(submission)
    output_dir_path = Path(output_dir)
    grade_jsonl(submission_path, output_dir_path, new_registry, normalize_metrics)


@cli.command("grade-sample")
@click.argument("submission", type=click.Path(exists=True))
@click.argument("competition_id", type=str)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the data will be stored.",
)
@click.option(
    "--normalize-metrics",
    is_flag=True,
    help="Normalize metrics to the range [0, 1] before saving results.",
)
def grade_sample(submission, competition_id, data_dir, normalize_metrics):
    """Grade a single sample (competition) in the eval."""
    new_registry = registry.set_data_dir(Path(data_dir))
    competition = new_registry.get_competition(competition_id)
    submission_path = Path(submission)
    report = grade_csv(submission_path, competition, normalize_metrics)
    logger.info("Competition report:")
    logger.info(json.dumps(report.to_dict(), indent=4))


@cli.command("generate")
@click.option(
    "-c",
    "--competition-id",
    help=f"ID of the competition to prepare. Valid options include: {', '.join(registry.list_competition_ids()[:5])}... (and more)",
    type=str,
    multiple=True,
    required=False,
)
@click.option("-a", "--all", is_flag=True, help="Generate all competitions.")
@click.option(
    "--lite",
    is_flag=True,
    help="Prepare all the low complexity competitions in lite version (MLE-bench Lite).",
)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(exists=True, file_okay=False),
    callback=validate_save_dir,
    required=True,
    help="Directory to save final results. Must be a direct subfolder of 'results/'.",
)
@click.option(
    "-pl",
    "--pipeline_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the pipeline to generate solutions",
)
@click.option(
    "-n",
    "--rounds",
    type=int,
    required=False,
    default=1,
    help="The number of rounds to run",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the data will be stored.",
)
@click.option(
    "--run-after", is_flag=True, help="Run workflows immediately after generation"
)
@click.option(
    "--submit-after",
    is_flag=True,
    help="Submit workflows immediately after running (requires --run-after)",
)
def generate(
    competition_id,
    all,
    lite,
    save_dir,
    pipeline_path,
    rounds,
    data_dir,
    run_after,
    submit_after,
):
    """Generate a competition."""
    if submit_after and not run_after:
        raise click.BadParameter("--calculate-after requires --run-after to be set")

    # Set the registry data directory to support lite competitions
    new_registry = registry.set_data_dir(Path(data_dir))

    async def process():
        await generate_and_run(
            competition_ids=competition_id,
            all=all,
            lite=lite,
            save_dir=save_dir,
            pipeline_path=pipeline_path,
            rounds=rounds,
            data_dir=data_dir,
            run_after=run_after,
            submit_after=submit_after
        )

    asyncio.run(process())


@cli.command("run")
@click.option(
    "-c",
    "--competition-id",
    help=f"ID of the competition to prepare. Valid options include: {', '.join(registry.list_competition_ids()[:5])}... (and more)",
    type=str,
    multiple=True,
    required=False,
)
@click.option("-a", "--all", is_flag=True, help="Generate all competitions.")
@click.option(
    "--lite",
    is_flag=True,
    help="Prepare all the low complexity competitions in lite version (MLE-bench Lite).",
)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(exists=True, file_okay=False),
    callback=validate_save_dir,
    required=True,
    help="Directory to save final results. Must be a direct subfolder of 'results/'.",
)
@click.option(
    "-n",
    "--rounds",
    type=int,
    required=False,
    default=1,
    help="The number of rounds to run",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the data will be stored.",
)
@click.option(
    "--submit-after", is_flag=True, help="Submit results immediately after running"
)
def run(competition_id, all, lite, save_dir, rounds, data_dir, submit_after):

    # Set the registry data directory to support lite competitions
    new_registry = registry.set_data_dir(Path(data_dir))

    async def process():
        await run_and_submit(
            competition_ids=competition_id,
            all=all,
            lite=lite,
            save_dir=save_dir,
            rounds=rounds,
            data_dir=data_dir,
            submit_after=submit_after,
            # registry=new_registry,
        )

    asyncio.run(process())


@cli.command("submit")
@click.option(
    "-c",
    "--competition-id",
    help=f"ID of the competition to prepare. Valid options include: {', '.join(registry.list_competition_ids()[:5])}... (and more)",
    type=str,
    multiple=True,
    required=False,
)
@click.option("-a", "--all", is_flag=True, help="Generate all competitions.")
@click.option(
    "--lite",
    is_flag=True,
    help="Prepare all the low complexity competitions in lite version (MLE-bench Lite).",
)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(exists=True, file_okay=False),
    callback=validate_save_dir,
    required=True,
    help="Directory to save final results. Must be a direct subfolder of 'results/'.",
)
@click.option(
    "-n",
    "--rounds",
    type=int,
    required=False,
    default=1,
    help="The number of rounds to run",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="tasks/mlebench",
    help="Path to the directory where the data will be stored.",
)
def submit(competition_id, all, lite, save_dir, rounds, data_dir):
    async def process():
        await submit_results(
            competition_ids=competition_id,
            all=all,
            lite=lite,
            save_dir=save_dir,
            rounds=rounds,
            data_dir=data_dir,
        )

    asyncio.run(process())


@cli.group("dev")
def dev():
    """Developer tools for extending MLE-bench."""


@dev.command("download-leaderboard")
@click.option(
    "-c",
    "--competition-id",
    help=f"Name of the competition to download the leaderboard for. Valid options include: {', '.join(registry.list_competition_ids()[:5])}... (and more)",
    type=str,
)
@click.option(
    "--all",
    is_flag=True,
    help="Download the leaderboard for all competitions.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force download the leaderboard, even if it already exists.",
)
def download_leaderboard(competition_id, all, force):
    """Download the leaderboard for a competition."""
    if all:
        for comp_id in registry.list_competition_ids():
            competition = registry.get_competition(comp_id)
            ensure_leaderboard_exists(competition, force=force)
    elif competition_id:
        competition = registry.get_competition(competition_id)
        ensure_leaderboard_exists(competition, force=force)
    else:
        raise click.BadParameter("Either --all or --competition-id must be specified.")


def main():
    cli()


if __name__ == "__main__":
    main()
