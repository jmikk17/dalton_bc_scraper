import re

import numpy as np
import pandas as pd

import auxil

# C6 xyz labels following the Orient format
C6_XYZ = {
    "00": 0,  # 00
    "0z": 1,  # 10
    "0x": 2,  # 11c
    "0y": 3,  # 11s
}


def extract_2nd_order_prop(content: str, wave_function: str, atomic_moment_order: int) -> dict:
    """Extract second order properties from Dalton output.

    Args:
        content (str): Content of the Dalton output file
        wave_function (str): The type of wave function used in the calculation, used for regex pattern selection
        atomic_moment_order (int): The atomic moment order of the calculation

    Returns:
        dict: Dictionary of the format {label: value} for each property

    """
    properties_00 = {}
    properties_0b = {}
    properties_ab = {}
    properties_a0 = {}

    if wave_function == "CC":
        pattern = (
            r"(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+([-+]?\d+\.\d+(?:E[-+]?\d+)?)"
        )
    else:
        pattern = r"@\s*-<<\s*(\w+)\s*;\s*(\w+)\s*>>\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)"

    for match in re.finditer(pattern, content):
        operator1 = match.group(1)
        operator2 = match.group(2)
        value = match.group(3)

        index1, _, xyz_comp1 = auxil.get_label(operator1)
        index2, _, xyz_comp2 = auxil.get_label(operator2)

        if xyz_comp1 == "00" and xyz_comp2 == "00":
            key = f"{index1}_{index2}"
            properties_00[key] = -float(value)
        if xyz_comp1 != "00" and xyz_comp2 == "00":
            key = f"{index1}:{xyz_comp1}_{index2}"
            properties_a0[key] = -float(value)
        if xyz_comp1 == "00" and xyz_comp2 != "00":
            key = f"{index1}_{index2}:{xyz_comp2}"
            properties_0b[key] = -float(value)
        if xyz_comp1 != "00" and xyz_comp2 != "00":
            key = f"{index1}:{xyz_comp1}_{index2}:{xyz_comp2}"
            properties_ab[key] = -float(value)

    if atomic_moment_order == 0:
        return {"00": properties_00}
    if wave_function == "CC":
        return {"00": properties_00, "0b": properties_0b, "a0": properties_a0, "ab": properties_ab}
    return {"00": properties_00, "0b": properties_0b, "ab": properties_ab}


def extract_1st_order_prop(content: str) -> list:
    """Extract MBIS charges from the Dalton output.

    Args:
        content (str): Content of the Dalton output file

    Returns:
        list: MBIS charges

    """
    mbis_pattern = r"MBIS converged!.*?Final converged results\s+Qatom(.*?)(?:\n\s*\n|\Z)"
    mbis_match = re.search(mbis_pattern, content, re.DOTALL)

    charge_list = []
    if mbis_match:
        charge_section = mbis_match.group(1)
        charge_pattern = r"\s+(\d+)\s+([-+]?\d+\.\d+)"
        charge_list.extend(
            [float(charge_match.group(2)) for charge_match in re.finditer(charge_pattern, charge_section)],
        )
    return charge_list


def read_2nd_order_prop(content: dict) -> pd.DataFrame:
    """Read the second order properties from dictionary into DataFrame with info from labels.

    Args:
        content (dict): Dictionary of parsed content from Dalton output file.

    Returns:
        pd.DataFrame: Dataframe with atm index, charge, xyz comp. and value.

    """
    parsed_data = []
    method = content["wave_function"]
    property_dict = content["2nd_order_properties"]
    for dict_type, inner_dict in property_dict.items():
        # loop over 00, 0b, a0, ab dicts
        for key, value in inner_dict.items():
            label_list = key.split("_")
            if dict_type == "00":
                index1 = label_list[0]
                index2 = label_list[1]
                xyz_comp1 = "00"
                xyz_comp2 = "00"
            if dict_type == "0b":
                index1 = label_list[0]
                xyz_comp1 = "00"
                idx_split = label_list[1].split(":")
                index2 = idx_split[0]
                xyz_comp2 = idx_split[1]
            if dict_type == "a0":
                index2 = label_list[1]
                xyz_comp2 = "00"
                idx_split = label_list[0].split(":")
                index1 = idx_split[0]
                xyz_comp1 = idx_split[1]
            if dict_type == "ab":
                idx_split1 = label_list[0].split(":")
                index1 = idx_split1[0]
                xyz_comp1 = idx_split1[1]
                idx_split2 = label_list[1].split(":")
                index2 = idx_split2[0]
                xyz_comp2 = idx_split2[1]

            parsed_data.append(
                {
                    "index1": int(index1),
                    "xyz1": xyz_comp1,
                    "index2": int(index2),
                    "xyz2": xyz_comp2,
                    "value": value,
                },
            )
            if method != "CC" and label_list[0] != label_list[1]:
                # CC prints all values, for other WF we manually add the other triangle of the matrix
                parsed_data.append(
                    {
                        "index2": int(index1),
                        "xyz2": xyz_comp1,
                        "index1": int(index2),
                        "xyz1": xyz_comp2,
                        "value": value,
                    },
                )

    return pd.DataFrame(parsed_data)


def extract_c6(content: str) -> dict:
    """Extract C6 coefficients from the Dalton output.

    Args:
        content (str): Content of the Dalton output file
        n_freq (int): Number of frequencies to extract

    Returns:
        dict: Dictionary of alpha(iomega) values

    Todo:
        - Logic for parsing needs to be double checked.

    """
    results = {}

    c6_pattern = r"(AM\w+)\s+(AM\w+)\s+([-]?\d+\.\d+)\n\s+GRIDSQ\s+ALPHA\n((?:\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\n){11})"

    for match in re.finditer(c6_pattern, content):
        operator1 = match.group(1)
        operator2 = match.group(2)

        data_block = match.group(4)

        index1, _, xyz_comp1 = auxil.get_label(operator1)
        index2, _, xyz_comp2 = auxil.get_label(operator2)

        key1 = f"{index1}_{index2}"
        key2 = f"{index2}_{index1}"
        reverse = False
        if key1 in results:
            key = key1
        elif key2 in results:
            key = key2
            reverse = True
        else:
            key = key1
            results[key1] = {}

        xyz_idx1 = C6_XYZ.get(xyz_comp1, 0)
        xyz_idx2 = C6_XYZ.get(xyz_comp2, 0)

        data_lines = data_block.strip().split("\n")
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                gridsq = -float(parts[0])
                alpha = float(parts[1])

                if str(gridsq) not in results[key]:
                    results[key][str(gridsq)] = np.zeros((4, 4))
                if reverse:
                    results[key][str(gridsq)][xyz_idx2, xyz_idx1] = alpha
                else:
                    results[key][str(gridsq)][xyz_idx1, xyz_idx2] = alpha
                if index1 == index2:
                    results[key][str(gridsq)][xyz_idx2, xyz_idx1] = alpha

    return results
