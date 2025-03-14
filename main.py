from __future__ import annotations

import sys
import warnings

import alpha
import auxil
import parse_calculation
import parse_coords
import parse_properties


def parse_dalton_output(content: str) -> None | dict:
    """Parse a Dalton output file and extract coordinates.

    Args:
        content (str): Content of Dalton output file

    Returns:
        dict: Dictionary containing parsed coordinates

    """
    main_dict = parse_calculation.extract_calculation_info(content)
    if not main_dict:
        sys.exit("Error: No calculation details found")

    coord_dict = parse_coords.extract_coordinates(content)
    if not coord_dict:
        sys.exit("Error: No coordinates found")
    charge_list = parse_properties.extract_1st_order_prop(content)
    if not charge_list:
        warnings.warn("No charges found", stacklevel=2)
        main_dict.update({"atoms": coord_dict})
    else:
        main_dict.update({"atoms": parse_coords.combine_coords_with_charges(coord_dict, charge_list)})

    property_dict = {
        "2nd_order_properties": parse_properties.extract_2nd_order_prop(
            content,
            main_dict["wave_function"],
            main_dict["atomic_moment_order"],
        ),
    }
    if not property_dict:
        sys.exit("Error: No second order properties found")
    main_dict.update(property_dict)

    return main_dict


def alpha_analysis(content: dict) -> dict:
    """Calculate alpha contributions from second order properties.

    Args:
        content (str): content of parsed JSON file

    Returns:
        dict: Dictionary containing alpha contributions

    """
    property_df = parse_properties.read_2nd_order_prop(content)

    geometry = parse_coords.read_coords(content)
    atmmom = content["atomic_moment_order"]

    alpha_dict = alpha.alpha_calc(property_df, geometry, atmmom)
    if alpha_dict is None:
        sys.exit("Error: Alpha analysis failed, dict is empty")

    return alpha.alpha_calc(property_df, geometry, atmmom)


def main() -> None:
    """Select between parsing and alpha analysis and write the output to a JSON file.

    Default option is to parse the Dalton output file and perform alpha analysis, but both can be selected separately.

    """
    args = auxil.setup_parser()

    if not (args.parse or args.alpha or args.all):
        args.all = True

    input_file, output_file = auxil.get_file_names(args)

    if not args.alpha:
        content = auxil.read_file(input_file, ".out")
        result = parse_dalton_output(content)
        if args.all:
            result.update(alpha_analysis(result))
    elif args.alpha:
        content = auxil.read_file(input_file, ".json")
        result = alpha_analysis(content)

    auxil.write_file(output_file, result)


if __name__ == "__main__":
    main()
