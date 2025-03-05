import re

import pandas as pd

import auxil


def extract_2nd_order_prop(content: str, wave_function: str) -> dict:
    """Extract second order properties from Dalton output.

    Args:
        content (str): Content of the Dalton output file
        wave_function (str): The type of wave function used in the calculation, used for regex pattern selection

    Returns:
        dict: Dictionary of the format {label: value} for each property

    """
    properties = {}

    if wave_function == "CC":
        pattern = (
            r"(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+([-+]?\d+\.\d+(?:E[-+]?\d+)?)"
        )
        for match in re.finditer(pattern, content):
            operator1 = match.group(1)
            operator2 = match.group(2)
            value = match.group(3)
            key = f"{operator1}_{operator2}"
            properties[key] = -float(value)
    else:
        pattern = r"@\s*-<<\s*(\w+)\s*;\s*(\w+)\s*>>\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)"
        matches = re.findall(pattern, content)

        for match in matches:
            operator1, operator2, value = match
            key = f"{operator1}_{operator2}"
            properties[key] = -float(value)

    return properties


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
    property_dict = content["properties"]
    for label, value in property_dict.items():
        label_list = label.split("_")
        index1, nuc_charge1, xyz_comp1 = auxil.get_label(label_list[0])
        index2, nuc_charge2, xyz_comp2 = auxil.get_label(label_list[1])
        parsed_data.append(
            {
                "index1": index1,
                "nuc_charge1": nuc_charge1,
                "xyz1": xyz_comp1,
                "index2": index2,
                "nuc_charge2": nuc_charge2,
                "xyz2": xyz_comp2,
                "value": value,
            },
        )
        if method != "CC" and label_list[0] != label_list[1]:
            # CC prints all values, for other WF we manually add the other triange of the matrix
            parsed_data.append(
                {
                    "index2": index1,
                    "nuc_charge2": nuc_charge1,
                    "xyz2": xyz_comp1,
                    "index1": index2,
                    "nuc_charge1": nuc_charge2,
                    "xyz1": xyz_comp2,
                    "value": value,
                },
            )

    return pd.DataFrame(parsed_data)
