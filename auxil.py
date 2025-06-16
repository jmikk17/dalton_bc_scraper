from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


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
        "-c",
        "--c6",
        action="store_true",
        help="Parse imaginary polarizability from Dalton output into JSON format",
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
        elif args.c6:
            args.output = path.with_suffix(".orient.txt")
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
            json.dump(content, file, indent=2, cls=NumpyEncoder)
    except OSError as e:
        sys.exit(f"Error writing file: {e}")


def write_c6(c6_dict: dict, labels: list, output_file: str = "c6_tmp.txt") -> None:
    """Write the C6 data in Orient format.

    Args:
        c6_dict (dict): Dictionary containing C6 data
        labels (list): List of atom labels
        output_file (str): Output file path

    """
    # To match Orient format, we need to have sorted frequencies as outer layer
    all_frequencies = set()
    for freq_data in c6_dict.values():
        all_frequencies.update([float(f) for f in freq_data])

    sorted_frequencies = sorted(all_frequencies, reverse=True)

    with Path(output_file).open("w") as file:
        for freq in sorted_frequencies:
            for key, freq_data in c6_dict.items():
                idx = key.split("_")
                idx1 = idx[0]
                idx2 = idx[1]
                site1 = labels[int(idx1) - 1]
                site2 = labels[int(idx2) - 1]

                freq_str = str(freq)
                found = False

                if freq_str in freq_data:
                    found = True
                else:
                    for f in freq_data:
                        if abs(float(f) - freq) < 1e-10:  # Allow for floating point precision issues
                            freq_str = f
                            found = True
                            break

                rank = 1

                if found:
                    matrix = freq_data[freq_str]

                    freq_formatted = "0.0000000E+00" if freq == 0.0 else f"{freq:.7E}"

                    if rank == 1:
                        orient_header = (
                            f"POL  SITE-LABELS  {site1}  {site2}  SITE-INDICES     {idx1}     {idx2}  "
                            f"RANK  0 :   1   BY     0 :   1   FREQ2  {freq_formatted}  CARTSPHER S"
                        )
                        file.write(orient_header + "\n")

                        for i in range(4):
                            row = ""
                            for j in range(4):
                                value_str = f"{matrix[i, j]:.7E}"
                                row += value_str + "   "

                            file.write(f"{row}\n")

                        file.write("END\n")
                    elif rank == 0:
                        orient_header = (
                            f"POL  SITE-LABELS  {site1}  {site2}  SITE-INDICES     {idx1}     {idx2}  "
                            f"RANK  0 :   0   BY     0 :   0   FREQ2  {freq_formatted}  CARTSPHER S"
                        )

                        file.write(orient_header + "\n")
                        file.write(f"{matrix[0, 0]:.7E}\n")
                        file.write("END\n")

                else:
                    sys.exit(f"Error: Frequency {freq} not found in C6 data for {site1} and {site2}")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""

    def default(self, obj: object) -> object:
        """Convert numpy data types to standard Python types.

        Args:
            obj (object): Object to be converted

        Returns:
            object: Converted object

        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
