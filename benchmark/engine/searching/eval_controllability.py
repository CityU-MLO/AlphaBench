import random
import os
import json
from datetime import datetime
from factors.lib.alpha158 import load_factors_alpha158
from agent.qlib_contrib.qlib_expr_parsing import FactorParser, print_tree
from agent.LLMExecutor import FactorGenerationQlibExecutor

from searcher.EAFactorSearcher import EAController
from searcher.TreeBasedSearcher import MCTSController


def print_tree_with_path(node, target_path_str, level=0, current_path=None):
    """
    Generate a string representation of a tree with markers indicating the target node 
    and the path leading to it.

    Markers:
        *  Target node
        +  Nodes along the path to the target node
           (including parent nodes of the target)
        (space) Other nodes not on the path to the target

    Parameters
    ----------
    node : object
        The current node to print. The node is expected to have:
        - a 'label' attribute (string) representing its name
        - a 'children' attribute (list) containing child nodes.
    
    target_path_str : str
        A string representing the target node's path in dot notation.
        For example, "1.0.2" means the path is:
        root -> child[1] -> child[0] -> child[2].
    
    level : int, optional
        The current depth level of the node in the tree. Used for indentation.
        Default is 0 (the root level).
    
    current_path : list of int, optional
        A list representing the current path from the root to this node.
        Default is None, which initializes it as an empty list.

    Returns
    -------
    str
        A string showing the tree structure with the appropriate markers.
    """
    if current_path is None:
        current_path = []

    # Convert target_path_str (e.g., "1.0.2") into a list of integers [1, 0, 2]
    target_path = list(map(int, target_path_str.split("."))) if target_path_str else []

    # Determine the marker for this node
    if current_path == target_path:
        marker = "*"
    elif current_path == target_path[: len(current_path)]:
        marker = "+"
    else:
        marker = " "

    # Build the line for this node, including indentation and marker
    line = f"{'  ' * level}{marker}{node.label}"

    # Collect lines for this subtree
    lines = [line]

    # Recursively process and append child nodes
    for idx, child in enumerate(node.children):
        child_str = print_tree_with_path(
            child, target_path_str, level + 1, current_path + [idx]
        )
        lines.append(child_str)

    return "\n".join(lines)


def generate_llm_prompt_MCTS(instruction):
    parent_tree_str = print_tree_with_path(
        instruction["parent_tree"], instruction["node_path"]
    )
    prompt = f"""You are part of a Monte Carlo Tree Search (MCTS) process for generating quantitative factor expressions.

At this step, the search has selected a specific subtree (located at node path {instruction['node_path']}) to expand.

Please:
- Extend or replace only the selected subtree at the given node path.
- Keep the remaining parts of the expression unchanged.
- Return a complete valid factor expression after this expansion.

Context:
Original factor name: {instruction['factor_name']}
Original factor expression:
{instruction['parent_expr']}

Original expression tree (with search path marked):
The '+' indicates nodes along the selected path, '*' marks the end point of exact target subtree:
{parent_tree_str}

Return the new factor expression below.
"""
    return prompt


def generate_llm_prompt_EA(instruction):
    if instruction["type"] == "mutation":
        parent_tree_str = print_tree_with_path(
            instruction["parent_tree"], instruction["target_node_path"]
        )
        prompt = f"""Please perform a mutation on the factor named {instruction['parent_id']}.
Target node path: {instruction['target_node_path']}

The + in expression tree is the path, and * is the end of path.

Original expression:
{instruction['parent_expr']}

Original expression tree:
{parent_tree_str}

Please modify the factor by mutating the subtree at the specified path.
Return the new factor expression.
"""
    elif instruction["type"] == "crossover":
        parent1_tree_str = print_tree_with_path(
            instruction["parent1_tree"], instruction["target_node_path1"]
        )
        parent2_tree_str = print_tree_with_path(
            instruction["parent2_tree"], instruction["target_node_path2"]
        )
        prompt = f"""Please perform a crossover between the two factors: {instruction['parent1_id']} and {instruction['parent2_id']}.
Target node path in {instruction['parent1_id']}: {instruction['target_node_path1']}
Target node path in {instruction['parent2_id']}: {instruction['target_node_path2']}

The + in expression tree is the path, and * is the end of path.

Original expression 1:
{instruction['parent1_expr']}

Original expression tree 1:
{parent1_tree_str}

Original expression 2:
{instruction['parent2_expr']}

Original expression tree 2:
{parent2_tree_str}

Please generate a new factor by combining the specified subtrees.
Return the new factor expression.
"""
    else:
        raise ValueError(f"Unknown instruction type: {instruction['type']}")

    return prompt


def test_control_generate_mcts(
    model="deepseek-chat",
    local_model=False,
    local_port=8000,
    save_dir="./run",
    use_offline_instructions=True,
    offline_instructions="./benchmark/data/mcts_factor_instruction.pkl",
    enable_cot=True,
):
    factor_parser = FactorParser()
    mcts_controller = MCTSController()

    executor = FactorGenerationQlibExecutor(max_threads=5, delay=0.1)
    result_dir = os.path.join(save_dir, "S5/MCTS")
    os.makedirs(result_dir, exist_ok=True)

    standard_factors, compile_factors = load_factors_alpha158(collection="kbar")

    mcts_factors_input = [
        {
            "name": name,
            "expr": expr.get("qlib_expression"),
            "parsed_tree": factor_parser.parse(expr.get("qlib_expression")),
        }
        for name, expr in compile_factors.items()
    ]

    all_prompts = []
    instruction_lists = []

    if use_offline_instructions:
        print("Loading offline MCTS instructions...")
        with open(offline_instructions, "rb") as f:
            import pickle

            instruction_lists = pickle.load(f)

        print(f"Load {len(instruction_lists)} instructions")
        for instruction in instruction_lists:
            prompts = generate_llm_prompt_MCTS(instruction)
            all_prompts.append(prompts)

    else:
        for factor in mcts_factors_input:
            print(f"Processing factor prompts: {factor['name']}")
            mcts_controller.init_root(factor["parsed_tree"], factor["expr"])
            instruction, node = mcts_controller.generate_instruction()
            instruction["factor_name"] = factor["name"]

            instruction_lists.append(instruction)
            prompts = generate_llm_prompt_MCTS(instruction)
            all_prompts.append(prompts)

    # import pdb;pdb.set_trace()
    with open(os.path.join(result_dir, "mcts_factor_instruction.pkl"), "wb") as f:
        import pickle

        pickle.dump(instruction_lists, f)

    print("Running searching test for MCTS...")
    result = executor.run(
        all_prompts,
        verbose=True,
        model=model,
        temperature=1.5,
        enable_cot=enable_cot,
        local=local_model,
        local_port=local_port,
    )
    with open(
        os.path.join(result_dir, "mcts_factor_result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def test_control_generate_ea(
    model="deepseek-chat",
    save_dir="./run",
    local_model=False,
    local_port=8000,
    use_offline_instructions=True,
    offline_instructions="./benchmark/data/ea_factor_instruction.pkl",
    num_instructions=100,
    enable_cot=True,
):
    factor_parser = FactorParser()

    executor = FactorGenerationQlibExecutor(max_threads=5, delay=0.1)

    result_dir = os.path.join(save_dir, "S5/EA")
    os.makedirs(result_dir, exist_ok=True)

    standard_factors, compile_factors = load_factors_alpha158()

    ea_factors_input = [
        {
            "name": name,
            "expr": expr.get("qlib_expression"),
            "parsed_tree": factor_parser.parse(expr.get("qlib_expression")),
        }
        for name, expr in compile_factors.items()
    ]

    ea_controller = EAController(mutation_rate=0.5, crossover_rate=0.5)
    if use_offline_instructions:
        print("Loading offline EA instructions...")
        with open(offline_instructions, "rb") as f:
            import pickle

            instructions_list = pickle.load(f)

        print(f"Load {len(instructions_list)} instructions")

    else:
        instructions_list = [
            ea_controller.generate_instruction(ea_factors_input)
            for _ in range(num_instructions)
        ]

    import pdb

    pdb.set_trace()
    print("Running searching test for EA...")
    instructions_promtps = [generate_llm_prompt_EA(ins) for ins in instructions_list]
    result = executor.run(
        instructions_promtps,
        verbose=True,
        model=model,
        temperature=1.5,
        enable_cot=enable_cot,
        local=local_model,
        local_port=local_port,
    )

    with open(os.path.join(result_dir, "ea_factor_instruction.pkl"), "wb") as f:
        import pickle

        pickle.dump(instructions_list, f)

    with open(
        os.path.join(result_dir, "improve_factor_result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # test_control_generate_mcts()
    test_control_generate_ea()
    # print("EA Controller test passed.")
