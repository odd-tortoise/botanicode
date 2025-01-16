from dataclasses import dataclass, field
from typing import Dict, Callable, Any, List
import numpy as np

from shapes import NodeShapeBlueprint


class NodeStateBlueprint:
    """Node state"""
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, value)

@dataclass
class NodeRuleBlueprint:
    """Rule for updating a nodes variables. They can act on the state or on the shape."""
    # this is for the ODEs
    dynamics:  Dict[str, Callable[[float, NodeStateBlueprint], Any]] = field(default_factory=dict)

    # this is for the derived variables, other relationships
    derived: Callable[[NodeStateBlueprint], Any] = None

    env_reading: List[str] = field(default_factory=list)

    init_rules: Dict[str, Any] = field(default_factory=dict)

    # qui anche le regole che agiscono sull'intero nodo -> bending, branching



class Model:
    def __init__(self, model_name):
        self.model_name = model_name if model_name else "default_model"
        self.shooting_rule = None
        self.nodes_blueprint = {}
        
        
    def add_blueprint(self, nodetype, state: NodeStateBlueprint, rules: NodeRuleBlueprint, shape: NodeShapeBlueprint):
        """
        Add a blueprint for a node type, linking its state structure and rules.

        Args:
            name (str): The unique name of the node blueprint (e.g., "Stem", "Leaf").
            state (NodeState): The template state for nodes of this type.
            rules (Rule): The rules that define how the node's state evolves.

        Raises:
            ValueError: If a blueprint with the same name already exists.
        """
        if nodetype in self.nodes_blueprint:
            raise ValueError(f"A blueprint for the type '{nodetype.__name__}' already exists.")

        # check that inside the state shape variables there are all the variables needed by the shape

        self.nodes_blueprint[nodetype] = {
            "state": state,
            "rules": rules,
            "shape": shape
        }

    def add_shooting_rule(self, rule):
        self.shooting_rule = rule
