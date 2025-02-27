from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_file_extension(file_path: str, ext: str) -> bool:
    """Check if the given path exists and has a valid extension.

    Args:
        file_path (str): Path to the file to check
        ext (str): Expected extension of the file

    Returns:
        bool: True if it's a valid extension and file exists, False otherwise

    """
    if not Path(file_path).is_file():
        return False

    return file_path.lower().endswith(ext)


def get_label(label: str) -> tuple[int, int, str]:
    """Get the atom info from the property label.

    Args:
        label (str): Property label

    Returns:
        list: Atom info in the format [index, nuc_charge, component]

    """
    index = int(label[2:4])
    nuc_charge = int(label[4:6])
    component = label[6:]
    return index, nuc_charge, component


def setup_parser() -> argparse.Namespace:
    """Set up the argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments

    """
    parser = argparse.ArgumentParser(description="Process dalton output files into JSON format")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-p",
        "--parse",
        action="store_true",
        help="Parse the Dalton output into JSON format",
    )
    group.add_argument(
        "-a",
        "--alpha",
        action="store_true",
        help="Perform alpha analysis on a JSON file and print to a new JSON file",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="(Default) Parse and perform alpha analysis, print to same JSON file",
    )
    parser.add_argument("input_file", help="Input file to process")
    parser.add_argument("-o", "--output", help="Output file (default: input_file.json)")

    return parser.parse_args()


def get_file_names(args: argparse.Namespace) -> tuple[str, str]:
    """Get the input and output file names from the arguments.

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        tuple: Input and output file names

    """
    if not args.output:
        path = Path(args.input_file)
        if args.parse or args.all:
            args.output = path.with_suffix(".json")
        elif args.alpha:
            args.output = path.with_suffix(".alpha.json")

    if (args.parse or args.all) and not check_file_extension(args.input_file, ".out"):
        sys.exit("Error: Input file must be a Dalton output file (.out)")
    elif args.alpha and not check_file_extension(args.input_file, ".json"):
        sys.exit("Error: Input file must be a JSON file (.json)")

    return args.input_file, args.output


def read_file(file_path: str, ext: str) -> str | dict:
    """Read the content of the file.

    Args:
        file_path (str): Path to the file to read
        ext (str): Extension of the file

    Returns:
        str: Content of the file

    """
    try:
        with Path(file_path).open("r") as file:
            if ext == ".out":
                return file.read()
            if ext == ".json":
                return json.load(file)
            sys.exit("Error: Invalid file extension")
    except FileNotFoundError:
        sys.exit(f"Error: File '{file_path}' not found")
    except OSError as e:
        sys.exit(f"Error reading file: {e}")


def write_file(file_path: str, content: dict) -> None:
    """Write the content to the file.

    Args:
        file_path (str): Path to the file to write
        content (dict): Content to dump to the file

    """
    try:
        with Path(file_path).open("w") as file:
            json.dump(content, file, indent=2)
    except OSError as e:
        sys.exit(f"Error writing file: {e}")
