import numpy as np
import random


class MCTSNode:
    def __init__(self, parent, path, expr_tree, expr_str):
        """
        MCTS search node.

        :param parent: Parent MCTSNode
        :param path: List[int], path to this node (e.g., [0,1,0])
        :param expr_tree: The FactorNode tree at this node
        :param expr_str: The expression string at this node
        """
        self.parent = parent
        self.path = path
        self.expr_tree = expr_tree
        self.expr_str = expr_str
        self.children = []
        self.visit_count = 0
        self.total_reward = 0.0


class MCTSController:
    def __init__(self):
        """
        MCTS controller for factor generation.

        """
        self.root = None

    def init_root(self, expr_tree, expr_str):
        """Initialize the root node of the search tree."""
        self.root = MCTSNode(None, [], expr_tree, expr_str)

    def select(self, node):
        """Select the child node using UCT."""
        if not node.children:
            return node

        def uct(n):
            if n.visit_count == 0:
                return float("inf")
            avg_reward = n.total_reward / n.visit_count
            exploration = 1.41 * np.sqrt(np.log(node.visit_count + 1) / n.visit_count)
            return avg_reward + exploration

        return max(node.children, key=uct)

    def expand(self, node):
        """Expand a new child at a random path."""
        new_path_str = self._random_node_path(node.expr_tree)
        new_path = self._path_to_list(new_path_str)
        child = MCTSNode(node, new_path, node.expr_tree, node.expr_str)
        node.children.append(child)
        return child

    def simulate(self, node):
        """Construct an instruction for LLM to generate factor expression."""
        instruction = {
            "type": "mcts",
            "node_path": ".".join(map(str, node.path)),
            "parent_expr": node.expr_str,
            "parent_tree": node.expr_tree,
        }
        return instruction

    def backpropagate(self, node, reward):
        """Backpropagate the reward up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def generate_instruction(self):
        """Run one MCTS step: select, expand, simulate, return instruction."""
        node = self.root
        while node.children:
            node = self.select(node)
        node = self.expand(node)
        instruction = self.simulate(node)
        return instruction, node

    def _random_node_path(self, tree, path=None):
        if path is None:
            path = []
        children = tree.children
        if not children:
            return ".".join(map(str, path))
        else:
            idx = random.randint(0, len(children) - 1)
            return self._random_node_path(children[idx], path + [idx])

    def _path_to_list(self, path_str):
        if not path_str:
            return []
        return list(map(int, path_str.split(".")))
