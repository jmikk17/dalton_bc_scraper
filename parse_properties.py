import re
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import pade

import auxil

# C6 xyz labels following the Orient format
XYZ_TO_SPHERICAL = {
    "00": 0,  # 00
    "0z": 1,  # 10
    "0x": 2,  # 11c
    "0y": 3,  # 11s
}

FREQ_SQ_LIST = [
    0.0000000e00,
    -4.3700000e-05,
    -1.3086000e-03,
    -9.1102000e-03,
    -3.9063200e-02,
    -1.3720890e-01,
    -4.5550980e-01,
    -1.5999700e00,
    -6.8604430e00,
    -4.7760340e01,
    -1.4306370e03,
]


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


def extract_imaginary(content: str) -> dict:
    """Extract alpha(i omega) from the Dalton output.

    Args:
        content (str): Content of the Dalton output file
        n_freq (int): Number of frequencies to extract

    Returns:
        dict: Dictionary of alpha(i omega) values. Outer keys are the labels of the atom pairs, inner keys are the
              frequencies, and values are the corresponding alpha(i omega) values.

    Todo:
        - Logic for parsing needs to be double checked.

    """
    results = {}

    imaginary_pattern = (
        r"(AM\w+)\s+(AM\w+)\s+([-]?\d+\.\d+)\n\s+GRIDSQ\s+ALPHA\n((?:\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\n){11})"
    )

    for match in re.finditer(imaginary_pattern, content):
        operator1 = match.group(1)
        operator2 = match.group(2)

        data_block = match.group(4)

        index1, _, xyz_comp1 = auxil.get_label(operator1)
        index2, _, xyz_comp2 = auxil.get_label(operator2)

        # We only have half the values, but in the new format they are not necessarily triangle of a mat
        # So we need to collect 1_2 and 2_1 terms in same mat
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

        xyz_idx1 = XYZ_TO_SPHERICAL.get(xyz_comp1, 0)
        xyz_idx2 = XYZ_TO_SPHERICAL.get(xyz_comp2, 0)

        data_lines = data_block.strip().split("\n")
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) >= 2:  # Ensure the line contains GRIDSQ and ALPHA values
                gridsq = -float(parts[0])
                alpha = float(parts[1])

                if str(gridsq) not in results[key]:
                    results[key][str(gridsq)] = np.zeros((4, 4))
                if reverse:
                    results[key][str(gridsq)][xyz_idx2, xyz_idx1] = alpha
                else:
                    results[key][str(gridsq)][xyz_idx1, xyz_idx2] = alpha
                if index1 == index2 and xyz_comp1 != xyz_comp2:
                    results[key][str(gridsq)][xyz_idx2, xyz_idx1] = alpha

    return results


def pade_approx(content: str) -> dict:
    """Extract Cauchy moments from the Dalton output, calculate alpha(i omega) using Pade approximations."""
    results = {}

    block_pattern = (
        r"\s*(AM\w+)\s+(AM\w+)\s+(-?\d+)\s+([-\d.Ee+]+)\s*\n"
        r"((?:\s+-?\d+\s+[-\d.Ee+]+\s*\n)+)"
    )

    for match in re.finditer(block_pattern, content):
        # The first line contains both labels and values, the rest is only data for Cauchy number n and value D_AB
        # Assumes coefficients starts from D(-4)=S(+2), and we need to start from D(0)=S(-2), D(n) = S(-n-2)
        operator1 = match.group(1)
        operator2 = match.group(2)
        _ = int(match.group(3))
        _ = float(match.group(4))
        data_block = match.group(5)

        n_list = []
        d_ab_list = []
        first = True

        for line in data_block.strip().splitlines():
            if first:
                first = False
                continue
            n, d_ab = line.split()
            n_list.append(int(n))
            d_ab_list.append(float(d_ab))

        print("Operator 1:", operator1, "Operator 2:", operator2)
        print(f"n_list: {n_list}")
        print(f"d_ab_list: {d_ab_list}")

        k = 10

        if len(n_list) < (k * 2 + 1):
            sys.exit(f"Not enough moments for Pade, need {k * 2 + 1} moments, got {len(n_list)}")
        p_low, q_low = pade(d_ab_list, n=k, m=(k - 1))
        p_high, q_high = pade(d_ab_list, n=k, m=k)

        for i in range(10):
            z_value = FREQ_SQ_LIST[i]

            pade_result_low = p_low(z_value) / q_low(z_value)
            pade_result_high = p_high(z_value) / q_high(z_value)

            normal_expansion = (
                d_ab_list[0] * z_value**0
                + d_ab_list[1] * z_value**1
                + d_ab_list[2] * z_value**2
                + d_ab_list[3] * z_value**3
                + d_ab_list[4] * z_value**4
            )
            print(
                "Omega sq:",
                FREQ_SQ_LIST[i],
                "Normal power series:",
                normal_expansion,
                "Pade approximation:",
                pade_result_low,
                pade_result_high,
            )

    sys.exit("Pade approximation not implemented yet")
