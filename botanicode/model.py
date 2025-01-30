from dataclasses import dataclass, field
from typing import Dict, Callable, Any, List
import numpy as np


from typing import List, Callable
import numpy as np

class Rule:
    def __init__(self, target_types: List[type], trainable=True, is_dynamic=False, no_params=4):
        """Initialize the Rule class."""
        self.target_types = target_types
        self.trainable = trainable
        self.is_dynamic = is_dynamic
        self.no_params = no_params
        self.params = np.zeros(no_params)
        self.action = None  # Action will be set later by the user

    def set_action(self, action: Callable):
        """Allow the user to set their custom action (apply function)."""
        self.action = action

    def apply(self, nodes: List[type]):
        """Apply the custom action to the nodes."""
        if self.action:
            return self.action(nodes, self.params)
        else:
            raise ValueError("Action is not defined. Use set_action() to define it.")



class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name if model_name else "default_model"
        
        self.plant_rules = []
        self.node_rules = []
        self.env_reads = {}
        
    def add_node_rule(self, rule):
        self.node_rules.append(rule)


    def get_trainable_params(self):
        """Return the trainable parameters of the model as one np.array."""
        trainable_params = []
        for rule in self.node_rules:
            if rule.trainable:
                trainable_params.append(rule.params)
        return np.concatenate(trainable_params)
    
    def set_trainable_params(self, params):
        """Set the trainable parameters of the model from a np.array, divide based on the rule no_params."""
        start = 0
        for rule in self.node_rules:
            if rule.trainable:
                end = start + rule.no_params
                rule.params = params[start:end]
                start = end
                
        
  
    def add_shooting_rule(self, rule):
        self.shooting_rule = rule

    def add_whole_plant_dynamic(self, var, node_types, ode):
        self.plant_dynamics.append({"var": var, "node_types": node_types, "ode": ode})

    def add_plant_state(self, state):
        self.plant_state = state

    def add_plant_rules(self, rules):
        self.plant_rules = rules