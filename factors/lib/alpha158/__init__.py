import os
import json

FACTOR_DIR = "./factors/lib/alpha158"
COMPILE_FILE = os.path.join(FACTOR_DIR, "qlib_compile_product.json")


def load_factors_alpha158_names():

    standard_factors = {}
    for file_name in os.listdir(FACTOR_DIR):
        if file_name.endswith(".json") and file_name != "qlib_compile_product.json":
            file_path = os.path.join(FACTOR_DIR, file_name)
            with open(file_path, "r") as f:
                factors = json.load(f)
                standard_factors[file_name.split(".")[0]] = factors

    return standard_factors


def load_factors_alpha158(exclude_var=None, collection=None):
    """
    Load standard factors and corresponding compiled factors,
    with optional exclusion of variables and optional collection selection.

    Parameters
    ----------
    exclude_var : str or None
        Keyword to exclude factors containing this variable in their expression.
    collection : str or None
        If specified, should be 'kbar', 'price', or 'rolling' to load that specific JSON file.
        If None, load all available factor JSON files.

    Returns
    -------
    standard_factors : list
        List of standard factor definitions.
    compile_factors : dict
        Dictionary of compiled factors corresponding to loaded standard factors.
    """
    # Load compiled factors
    with open(COMPILE_FILE, "r") as f:
        compile_factors_raw = json.load(f)
        compile_factors_all = {item["name"]: item for item in compile_factors_raw}

    # Load standard factors
    standard_factors = []

    if collection is not None:
        if isinstance(collection, str):
            target_file = f"{collection}.json"
            file_path = os.path.join(FACTOR_DIR, target_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    factors = json.load(f)
                    standard_factors.extend(factors)
            else:
                print(
                    f"⚠️ Warning: Collection file '{target_file}' does not exist in {FACTOR_DIR}."
                )
        elif isinstance(collection, list):
            for coll in collection:
                target_file = f"{coll}.json"
                file_path = os.path.join(FACTOR_DIR, target_file)
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        factors = json.load(f)
                        standard_factors.extend(factors)
                else:
                    print(
                        f"⚠️ Warning: Collection file '{target_file}' does not exist in {FACTOR_DIR}."
                    )
        else:
            raise ValueError("Collection must be a string or a list of strings.")
    else:
        for file_name in os.listdir(FACTOR_DIR):
            if file_name.endswith(".json") and file_name != "qlib_compile_product.json":
                file_path = os.path.join(FACTOR_DIR, file_name)
                with open(file_path, "r") as f:
                    factors = json.load(f)
                    standard_factors.extend(factors)

    # Exclude factors containing exclude_var if specified
    if exclude_var:
        # import pdb;pdb.set_trace()
        standard_factors = [
            factor
            for factor in standard_factors
            if exclude_var not in factor.get("expression", {}).get("template", "")
        ]

    # Keep only compiled factors corresponding to the loaded standard factors
    standard_names = {factor["name"] for factor in standard_factors}
    compile_factors = {
        name: item
        for name, item in compile_factors_all.items()
        if name in standard_names
    }

    # Further exclude compiled factors if exclude_var is specified
    if exclude_var:
        compile_factors = {
            name: item
            for name, item in compile_factors.items()
            if exclude_var not in item.get("qlib_expression", "")
        }

    return standard_factors, compile_factors


if __name__ == "__main__":

    # Example usage
    standard_factors, compile_factors = load_factors_alpha158(exclude_var="vwap")
    print(
        f"Loaded {len(standard_factors)} standard factors and {len(compile_factors)} compiled factors."
    )
    print("Standard Factors:", standard_factors[:2])  # Print first 2 for brevity
    print(
        "Compiled Factors:", list(compile_factors.keys())[:2]
    )  # Print first 2 keys for brevity

    standard_factors, compile_factors = load_factors_alpha158(collection="kbar")
    print(
        f"Loaded {len(standard_factors)} standard factors from 'kbar' collection and {len(compile_factors)} compiled factors."
    )
    print(
        "Standard Factors from 'kbar':", standard_factors[:2]
    )  # Print first 2 for brevity
