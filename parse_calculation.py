import re
import sys


def extract_calculation_info(content: str) -> dict:
    """Extract calculation type information from Dalton output.

    Args:
        content (str): The content of the Dalton output file

    Returns:
        dict: Dictionary containing calculation type information

    """
    result = {}

    atmmom_match = re.search(r"\.ATMMOM\s*\n\s*(\d+)", content, re.MULTILINE)
    if atmmom_match:
        result["atomic_moment_order"] = int(atmmom_match.group(1))
    else:
        sys.exit("Error: No atomic moment order found")

    result.update(extract_wave_function_type(content))
    result.update(extract_wave_function_info(content, result.get("wave_function")))
    result.update(extract_other_info(content))

    return result


def extract_wave_function_type(content: str) -> dict:
    """Extract wave function type from Dalton output.

    Args:
        content (str): The content of the Dalton output file

    Returns:
        dict: Dictionary containing wave function type

    """
    result = {}
    wf_match = re.search(r"@\s*Wave function type\s*---\s*([A-Za-z0-9\(\)-]+)\s*---", content)
    if wf_match:
        result["wave_function"] = wf_match.group(1).strip()
    if result:
        return result
    sys.exit("Error: No wave function type found")


def extract_wave_function_info(content: str, wave_function: str) -> dict:
    """Extract additional information based on the wave function type.

    Args:
        content (str): The content of the Dalton output file
        wave_function (str): The type of wave function used in the calculation

    Returns:
        dict: Dictionary containing additional wave function information

    """
    result = {}
    if wave_function == "KS-DFT":
        dft_match = re.search(r"This is a DFT calculation of type:\s*([A-Za-z0-9-]+)", content)
        if dft_match:
            result["wave_function_type"] = dft_match.group(1).strip()
        else:
            sys.exit("Error: No DFT type found")
    elif wave_function == "CC":
        cc_model_match = re.search(r"Coupled Cluster model is:\s*([A-Za-z0-9]+)", content)
        if cc_model_match:
            result["wave_function_type"] = cc_model_match.group(1).strip()
        else:
            sys.exit("Error: No CC model found")
    elif wave_function == "MC-SCF":
        inactive_match = re.search(r"@\s*Inactive orbitals\s+(\d+)\s*\|\s*(\d+)", content)
        if inactive_match:
            result["inactive_orbitals"] = int(inactive_match.group(1))
        else:
            sys.exit("Error: No inactive orbitals found")
        electrons_match = re.search(r"@\s*Number of electrons in active shells\s+(\d+)", content)
        active_match = re.search(r"@\s*Active orbitals\s+(\d+)\s*\|\s*(\d+)", content)
        if electrons_match and active_match:
            result["active_space"] = (int(electrons_match.group(1)), int(active_match.group(1)))
        else:
            sys.exit("Error: No active space found")
    return result


def extract_other_info(content: str) -> dict:
    """Extract basis set and total charge from Dalton output.

    Args:
        content (str): The content of the Dalton output file

    Returns:
        dict: Dictionary containing additional calculation information

    """
    result = {}
    basis_match = re.search(r'Basis set used is "([^"]+)" from the basis set library', content)
    if basis_match:
        result["basis"] = basis_match.group(1).strip()
    else:
        sys.exit("Error: No basis set found")
    charge_match = re.search(r"@\s*Total charge of the molecule\s+(-?\d+)", content)
    if charge_match:
        result["total_charge"] = int(charge_match.group(1))
    else:
        sys.exit("Error: No total charge of molecule found")
    return result
