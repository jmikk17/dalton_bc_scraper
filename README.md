# Dalton Output Parser

A Python tool for extracting and analyzing data from output files from the crk-oslo feature branch of the Dalton fork (https://github.com/jmikk17/dalton_bc).

## Description

This tool parses Dalton output files to extract various properties including:
- Calculation information (wave function type, basis set)
- Coordinates
- Population analysis (MBIS)
- Distributed polarizabilities (bond capacities)
- Polarizability calculations

## Installation
Requires NumPy and Pandas.
```bash
git clone https://github.com/jmikk17/dalton_bc_scraper.git
```

## Usage
The following command preforms the parsing and analysis and saves it to "example.json":
```bash
python3 main.py example.out
```

## Project Structure
* main.py: Main entry point and orchestration
* parse_calculation.py: Extracts calculation details (wave function, basis set)
* parse_coords.py: Extracts atomic coordinates
* parse_properties.py: Extracts properties (MBIS charges, second-order properties)
* alpha.py: Handles polarizability analysis
* auxil.py: Auxiliary helper functions
