import sys
import click
from importlib.resources import as_file, files

from pocketmidi.humanise import load_profile, humanise


@click.command()
@click.argument("input_path",  type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option("--genre",     default="rock",  show_default=True,
              help="Genre profile to use.")
@click.option("--intensity", default=1.0,     show_default=True,
              type=click.FloatRange(0.0, 1.0),
              help="Humanisation strength 0.0–1.0.")
@click.option("--section",   default="beat",  show_default=True,
              type=click.Choice(["beat", "fill"]),
              help="Beat section type.")
@click.option("--seed",      default=None,    type=int,
              help="Random seed for reproducibility.")
@click.option("--verbose",   is_flag=True,
              help="Log per-hit fallback level.")
def main(input_path, output_path, genre, intensity, section, seed, verbose):
    """Humanise programmed drum MIDI using real drummer performance data."""
    resource = files("pocketmidi.profiles").joinpath(f"{genre}.json")
    try:
        with as_file(resource) as profile_path:
            profiles = load_profile(profile_path)
    except FileNotFoundError:
        click.echo(f"Error: no profile found for genre '{genre}'.", err=True)
        sys.exit(1)

    try:
        humanise(
            input_path=input_path,
            output_path=output_path,
            profiles=profiles,
            genre=genre,
            beat_type=section,
            intensity=intensity,
            seed=seed,
            verbose=verbose,
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
