import random


class EAController:
    def __init__(self, mutation_rate, crossover_rate):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def generate_instruction(self, population):
        op_type = random.choices(
            ["mutation", "crossover"],
            weights=[self.mutation_rate, self.crossover_rate],
            k=1,
        )[0]

        if op_type == "mutation":
            parent = random.choice(population)
            path = self._random_node_path(parent["parsed_tree"])
            instruction = {
                "type": "mutation",
                "parent_id": parent["name"],
                "parent_expr": parent["expr"],
                "parent_tree": parent["parsed_tree"],
                "target_node_path": path,
            }
        else:
            p1, p2 = random.sample(population, 2)
            path1 = self._random_node_path(p1["parsed_tree"])
            path2 = self._random_node_path(p2["parsed_tree"])
            instruction = {
                "type": "crossover",
                "parent1_id": p1["name"],
                "parent1_expr": p1["expr"],
                "parent1_tree": p1["parsed_tree"],
                "parent2_id": p2["name"],
                "parent2_expr": p2["expr"],
                "parent2_tree": p2["parsed_tree"],
                "target_node_path1": path1,
                "target_node_path2": path2,
            }

        return instruction

    def _random_node_path(self, tree, path=None):
        if path is None:
            path = []
        children = tree.children
        if not children:
            return ".".join(map(str, path))
        else:
            idx = random.randint(0, len(children) - 1)
            return self._random_node_path(children[idx], path + [idx])
