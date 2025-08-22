import re
from zss import Node
import ast


class FactorNode(Node):
    def __init__(self, label):
        super().__init__(label)
        self.label = label
        self._children = []

    def addkid(self, node):
        self._children.append(node)
        return super().addkid(node)

    def get_children(self):
        return self._children


class FactorParser:
    def __init__(self):
        self.param_map = {}
        self.param_counter = 1
        self.var_map = {}
        self.var_counter = 1

    def preprocess(self, expr):
        # Replace variables like $close
        def replace_var(match):
            var = match.group()
            safe = f"var{self.var_counter}"
            self.var_map[safe] = var
            self.var_counter += 1
            return safe

        expr = re.sub(r"\$\w+", replace_var, expr)

        # Replace parameters like {lag}
        def replace_param(match):
            param = match.group()
            if param not in self.param_map:
                cname = f"C{self.param_counter}"
                self.param_map[param] = cname
                self.param_counter += 1
            return self.param_map[param]

        expr = re.sub(r"\{\w+\}", replace_param, expr)
        return expr

    def parse(self, expr):
        pre_expr = self.preprocess(expr)
        tree = ast.parse(pre_expr, mode="eval")
        return self._convert(tree.body)

    def _convert(self, node):
        if isinstance(node, ast.BinOp):
            op_name = self._get_op_name(node.op)
            root = FactorNode(op_name)
            root.addkid(self._convert(node.left))
            root.addkid(self._convert(node.right))
            return root
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            root = FactorNode(func_name)
            for arg in node.args:
                root.addkid(self._convert(arg))
            return root
        elif isinstance(node, ast.Name):
            if node.id in self.var_map:
                return FactorNode(self.var_map[node.id])
            elif node.id in self.param_map.values():
                return FactorNode(node.id)
            else:
                return FactorNode(node.id)
        elif isinstance(node, ast.Constant):
            return FactorNode(str(node.value))
        elif isinstance(node, ast.UnaryOp):
            op_name = self._get_op_name(node.op)
            root = FactorNode(op_name)
            root.addkid(self._convert(node.operand))
            return root
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only simple comparisons supported")
            op_name = self._get_op_name(node.ops[0])
            root = FactorNode(op_name)
            root.addkid(self._convert(node.left))
            root.addkid(self._convert(node.comparators[0]))
            return root
        else:
            raise ValueError(f"Unsupported AST node: {node}")

    def _get_op_name(self, op):
        if isinstance(op, ast.Add):
            return "Add"
        elif isinstance(op, ast.Sub):
            return "Sub"
        elif isinstance(op, ast.Mult):
            return "Mul"
        elif isinstance(op, ast.Div):
            return "Div"
        elif isinstance(op, ast.USub):
            return "Neg"
        elif isinstance(op, ast.Gt):
            return "Gt"
        elif isinstance(op, ast.Lt):
            return "Lt"
        elif isinstance(op, ast.GtE):
            return "GtE"
        elif isinstance(op, ast.LtE):
            return "LtE"
        elif isinstance(op, ast.Eq):
            return "Eq"
        elif isinstance(op, ast.NotEq):
            return "NotEq"
        else:
            raise ValueError(f"Unsupported operator: {op}")

    # --- New Feature: Complexity Analysis ---
    def get_complexity(self, root: FactorNode):
        stats = {
            "node_count": 0,
            "depth": 0,
            "operator_count": 0,
            "function_count": 0,
            "var_count": 0,
            "param_count": 0,
        }

        vars_seen, params_seen = set(), set()

        def traverse(node, depth=1):
            stats["node_count"] += 1
            stats["depth"] = max(stats["depth"], depth)

            if node.label in [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Neg",
                "Gt",
                "Lt",
                "GtE",
                "LtE",
                "Eq",
                "NotEq",
            ]:
                stats["operator_count"] += 1
            elif node.label.startswith("C"):  # parameter
                params_seen.add(node.label)
            elif node.label.startswith("var"):  # variable
                vars_seen.add(node.label)
            elif (
                node.get_children()
            ):  # function call (if has children and not an operator)
                stats["function_count"] += 1

            for child in node.get_children():
                traverse(child, depth + 1)

        traverse(root)

        stats["var_count"] = len(vars_seen)
        stats["param_count"] = len(params_seen)

        # Composite score (simple heuristic)
        stats["complexity_score"] = (
            stats["node_count"]
            + 2 * stats["operator_count"]
            + 2 * stats["function_count"]
            + stats["depth"]
        )

        return stats


def print_tree(node, level=0):
    """
    Nicely print the tree.
    """
    print("  " * level + node.label)
    for child in node.children:
        print_tree(child, level + 1)


# Example usage
if __name__ == "__main__":
    expressions = [
        "($close - $open) / $open",
        "(Less($open, $close) - $low) / $open",
        "Quantile($close, {window}, 0.8) / $close",
        "Mul($close, $volume)",
        "If($close > $open, $close - $open, 0)",
    ]

    parser = FactorParser()
    for expr in expressions:
        print(f"\n✅ Parsing: {expr}")
        tree = parser.parse(expr)
        print_tree(tree)
        print(f"Parameter map: {parser.param_map}")

    expr1 = "Mul(Rank(Sub($close, Ref($close, {lag})), {price_rank_window}), Rank(Div($volume, Mean($volume, {vol_window})), {volume_rank_window}))"
    expr2 = "($close - $open) / $open"

    parser1 = FactorParser()
    tree1 = parser1.parse(expr1)

    parser2 = FactorParser()
    tree2 = parser2.parse(expr2)

    print("✅ Parsed Tree 1:")
    print_tree(tree1)
    print(f"Parameter map: {parser1.param_map}")

    print("\n✅ Parsed Tree 2:")
    print_tree(tree2)
    print(f"Parameter map: {parser2.param_map}")

    # Compute tree edit distance
    from zss import simple_distance

    distance = simple_distance(
        tree1,
        tree2,
        get_children=lambda node: node.children,
        get_label=lambda node: node.label,
    )
    print(f"\nTree Edit Distance: {distance}")
