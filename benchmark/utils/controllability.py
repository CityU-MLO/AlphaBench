from agent.qlib_contrib.qlib_expr_parsing import FactorParser, print_tree


def check_MCTS_controlled_factor_output(
    factor_expr, origin_expr, instruction, verbose=False
):
    parser = FactorParser()
    generated_formula = factor_expr["expression"]["template"]
    origin_formula = origin_expr["expression"]["template"]

    generated_ast = parser.parse(generated_formula)
    origin_ast = parser.parse(origin_formula)

    instruction_paths = instruction["node_path"]  # e.g. ['0.0', '1', '0.0.1']
    # operator_budget = instruction["operator_budget"]

    # Helper: get subtree at index path
    def get_subtree_by_index_path(root, index_path_str):
        index_path_str = index_path_str.strip(".")
        if index_path_str == "":
            return root
        indices = [int(x) for x in index_path_str.split(".") if x != ""]
        current = root
        for idx in indices:
            children = current.get_children()
            if idx < 0 or idx >= len(children):
                return None
            current = children[idx]
        return current

    # Helper: subtree string
    def subtree_to_str(node):
        if not node.get_children():
            return node.label
        return (
            f"{node.label}("
            + ",".join([subtree_to_str(k) for k in node.get_children()])
            + ")"
        )

    # Helper: count operators
    def count_operators(node):
        if node.label.startswith("var") or node.label.startswith("C"):
            cnt = 0
        else:
            cnt = 1
        for k in node.get_children():
            cnt += count_operators(k)
        return cnt

    modified = False
    path_valid = True

    for path in instruction_paths:
        origin_subtree = get_subtree_by_index_path(origin_ast, path)
        generated_subtree = get_subtree_by_index_path(generated_ast, path)

        if origin_subtree is None:
            if verbose:
                print(f"Original factor: node not found at path {path}")
            path_valid = False
            continue
        if generated_subtree is None:
            if verbose:
                print(f"Generated factor: node not found at path {path}")
            path_valid = False
            continue

        origin_str = subtree_to_str(origin_subtree)
        gen_str = subtree_to_str(generated_subtree)

        if origin_str != gen_str:
            if verbose:
                print(f"Subtree at path {path} has been modified.")
            modified = True
        else:
            if verbose:
                print(f"Subtree at path {path} has not been modified.")

    operator_count = count_operators(generated_ast)

    if verbose:
        print(f"Operator count: {operator_count}")

    passed = modified and path_valid

    if passed:
        print("Factor meets control requirements.")
    else:
        print("Factor does not meet control requirements.")
        print(f"Modified: {modified}, Path valid: {path_valid}")

    return passed


def check_EA_controlled_factor_output(child_expr, instruction, verbose=False):
    if child_expr is None:
        if verbose:
            print("Child expression is None, cannot perform EA control check.")
        return False

    parser = FactorParser()
    child_ast = parser.parse(child_expr["expression"]["template"])

    def get_subtree_by_index_path(root, index_path_str):
        index_path_str = index_path_str.strip(".")
        if index_path_str == "":
            return root
        indices = [int(x) for x in index_path_str.split(".") if x != ""]
        current = root
        for idx in indices:
            children = current.get_children()
            if idx < 0 or idx >= len(children):
                return None
            current = children[idx]
        return current

    def subtree_to_str(node):
        if not node.get_children():
            return node.label
        return (
            f"{node.label}("
            + ",".join([subtree_to_str(k) for k in node.get_children()])
            + ")"
        )

    modified = False
    path_valid = True
    op_type = instruction["type"]

    if op_type == "mutation":
        parent_ast = instruction["parent_tree"]
        path = instruction["target_node_path"]

        split_path = path.strip(".").split(".")
        path_valid = True
        modified = False

        # Check each prefix path: 1, 1.0, 1.0.1, ...
        for i in range(1, len(split_path) + 1):
            sub_path = ".".join(split_path[:i])
            parent_sub = get_subtree_by_index_path(parent_ast, sub_path)
            child_sub = get_subtree_by_index_path(child_ast, sub_path)

            if parent_sub is None or child_sub is None:
                path_valid = False
                if verbose:
                    print(f"Invalid subtree at path: {sub_path}")
                break  # Can't proceed further
            elif subtree_to_str(parent_sub) != subtree_to_str(child_sub):
                modified = True
                if verbose:
                    print(f"Detected modification at path: {sub_path}")
                break  # Early exit on first valid diff

        if not modified and verbose:
            print(f"No modification detected along path: {path}")

    elif op_type == "crossover":
        p1_ast = instruction["parent1_tree"]
        p2_ast = instruction["parent2_tree"]
        path1 = instruction["target_node_path1"]
        path2 = instruction["target_node_path2"]

        p1_subtree = get_subtree_by_index_path(p1_ast, path1)
        p2_subtree = get_subtree_by_index_path(p2_ast, path2)
        c1_subtree = get_subtree_by_index_path(child_ast, path1)
        c2_subtree = get_subtree_by_index_path(child_ast, path2)

        if None in [p1_subtree, p2_subtree, c1_subtree, c2_subtree]:
            path_valid = False
            if verbose:
                print("One or more crossover paths invalid")
        else:
            match1 = subtree_to_str(p1_subtree) == subtree_to_str(c1_subtree)
            match2 = subtree_to_str(p2_subtree) == subtree_to_str(c2_subtree)
            if match1 or match2:
                modified = True
                if verbose:
                    print(
                        "Crossover subtrees match parent1 and parent2 at specified paths"
                    )
            else:
                if verbose:
                    print("Crossover mismatch at one or more paths")

    passed = modified and path_valid

    if verbose:
        print("Passed EA control check" if passed else "Failed EA control check")
        print(f"Modified: {modified}, Path valid: {path_valid}")

    return passed
