import numpy as np
import pandas as pd


def alpha_calc(df: pd.DataFrame, geo: list, atmmom: int) -> dict:
    """Calculate alpha contributions from second order properties.

    Args:
        df (pd.DataFrame): DataFrame containing parsed label info and second order properties
        geo (list): Coordinates of the atoms
        atmmom (int): Variable describing the atomic moment order of the calculation

    Returns:
        dict: Dictionary with lists of lists containing 3by3 matrices of alpha contributions

    """
    alpha_00 = np.zeros((3, 3))
    alpha_0b = np.zeros((3, 3))
    alpha_a0 = np.zeros((3, 3))
    alpha_ab = np.zeros((3, 3))

    for _index, row in df.iterrows():
        val = row["value"]
        if row["xyz1"] == "00" and row["xyz2"] == "00":
            alpha_00 = update_alpha(alpha_00, val, row, geo, a_type="00")
            if atmmom == 0:
                continue
        elif row["xyz1"] == "00":
            alpha_0b = update_alpha(alpha_0b, val, row, geo, a_type="0b")
        elif row["xyz2"] == "00":
            alpha_a0 = update_alpha(alpha_a0, val, row, geo, a_type="a0")
        else:
            alpha_ab = update_alpha(alpha_ab, val, row, geo, a_type="ab")

    return format_alpha_result(alpha_00, alpha_0b, alpha_a0, alpha_ab, atmmom)


def update_alpha(alpha: np.ndarray, val: float, row: pd.Series, geo: list, a_type: str) -> np.ndarray:
    """Calculate and add alpha contribution to the alpha matrix depending on type of calculation.

    Args:
        alpha (np.ndarray): The 3by3 alpha matrix to be updated
        val (float): Value of the second order property
        row (pd.Series): Row of the DataFrame containing the label info and value
        geo (list): Coordinates of the atoms
        a_type (str): Type of alpha matrix to be updated

    Returns:
        np.ndarray: Updated alpha matrix

    """
    if a_type == "00":
        coord1 = geo[row["index1"] - 1]
        coord2 = geo[row["index2"] - 1]
        for i in range(3):
            for j in range(3):
                alpha[i, j] += val * coord1[i] * coord2[j]
    elif a_type == "0b":
        coord = geo[row["index1"] - 1]
        comp = get_component(row["xyz2"])
        for i in range(3):
            alpha[i, comp] += val * coord[i]
    elif a_type == "a0":
        coord = geo[row["index2"] - 1]
        comp = get_component(row["xyz1"])
        for i in range(3):
            alpha[comp, i] += val * coord[i]
    elif a_type == "ab":
        comp1 = get_component(row["xyz1"])
        comp2 = get_component(row["xyz2"])
        alpha[comp1, comp2] += val
    return alpha


def get_component(xyz: str) -> int:
    """Get xyz component index from label.

    Args:
        xyz (str): "00", "0x", "0y" or "0z"

    Returns:
        int: index of the component

    """
    components = {"0x": 0, "0y": 1, "0z": 2}
    return components.get(xyz)


def format_alpha_result(
    alpha_00: np.ndarray,
    alpha_0b: np.ndarray,
    alpha_a0: np.ndarray,
    alpha_ab: np.ndarray,
    atmmom: int,
) -> dict:
    """Format the alpha matrices into a dictionary, and create a sum entry.

    Args:
        alpha_00 (np.ndarray): Alpha matrix for 00 contributions
        alpha_0b (np.ndarray): for 0b contributions
        alpha_a0 (np.ndarray): for a0 contributions
        alpha_ab (np.ndarray): for ab contributions
        atmmom (int): Atomic moment order

    Returns:
        dict: Formatted dictionary with alpha matrices

    """
    alpha_tot = alpha_00 + alpha_0b + alpha_a0 + alpha_ab
    if atmmom == 0:
        return {"polarizability": {"alpha_00": (-alpha_00).tolist()}}
    return {
        "polarizability": {
            "alpha_00": (-alpha_00),
            "00_iso_fraction": np.round(np.trace(-alpha_00) / np.trace(-alpha_tot), decimals=3),
            "alpha_0b": (-alpha_0b),
            "0b_iso_fraction": np.round(np.trace(-alpha_0b) / np.trace(-alpha_tot), decimals=3),
            "alpha_a0": (-alpha_a0),
            "a0_iso_fraction": np.round(np.trace(-alpha_a0) / np.trace(-alpha_tot), decimals=3),
            "alpha_ab": (-alpha_ab),
            "ab_iso_fraction": np.round(np.trace(-alpha_ab) / np.trace(-alpha_tot), decimals=3),
            "alpha_tot": (-(alpha_tot)),
        },
    }
