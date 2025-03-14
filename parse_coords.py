import re
import sys


def extract_coordinates(content: str) -> list[dict[str, float]]:
    """Extract Cartesian coordinates from Dalton output.

    For larger molecules, this approach requires that the condition in subroutine pricar.F is commented out,
    since Cartesian coordinates are not printed by default. Can also be enabled by print level in Dalton input.

    Args:
        content (str): The content of the Dalton output file

    Returns:
        list: List of dictionaries containing atom labels and coordinates

    """
    coord_section = re.search(
        r"Cartesian Coordinates \(a\.u\.\)\s*\n\s*-+\s*\n\s*Total number of coordinates:\s*(\d+)"
        r"([\s\S]+?)(?:\n\s*\n|\n\s*Interatomic)",
        content,
    )

    if not coord_section:
        sys.exit("Error: No coordinates found")

    coord_text = coord_section.group(2)

    coords = []

    coord_pattern = (
        r"([A-Za-z0-9]+)\s*:\s*(\d+)\s*x\s*([-+]?\d*\.\d+)\s*(\d+)\s*y\s*([-+]?\d*\.\d+)\s*(\d+)\s*z\s*([-+]?\d*\.\d+)"
    )

    for i, match in enumerate(re.finditer(coord_pattern, coord_text)):
        atom_label_with_number = match.group(1)
        atom_label = re.match(r"([A-Za-z]+)", atom_label_with_number).group(1)
        x = float(match.group(3))
        y = float(match.group(5))
        z = float(match.group(7))

        coords.append({"label": atom_label, "index": i + 1, "x": x, "y": y, "z": z})

    return coords


def combine_coords_with_charges(coords: dict, charges: list) -> dict:
    """Insert charges into the coordinate dictionary.

    Args:
        coords (dict): Dictionary containing atom labels and coordinates
        charges (list): List of charges

    Returns:
        dict: Dictionary containing atom labels, coordinates and charges

    """
    for i, atom in enumerate(coords):
        if "MBIS charge" not in atom:
            atom["MBIS charge"] = charges[i]

    return coords


def read_coords(content: str) -> list:
    """Extract coordinates from JSON file.

    Args:
        content (str): Content of the JSON file

    Returns:
        list: Coordinates of the molecule

    """
    return [(atom["x"], atom["y"], atom["z"]) for atom in content["atoms"]]
