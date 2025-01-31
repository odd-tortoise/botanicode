from dataclasses import dataclass, field
from typing import Dict, Callable, Any, List
import numpy as np


from typing import List, Callable
import numpy as np

class Rule:
    def __init__(self, is_dynamic=False, trainable=True, no_params=4):
        """Initialize the Rule class."""
        self.trainable = trainable
        self.no_params = no_params
        self.is_dynamic = is_dynamic
        self.params = np.zeros(no_params)
        self.action = None  # Action will be set later by the user
        self.var = None

    def set_action(self, action: Callable, var : str = None, types : List[type] = None):
        """Allow the user to set their custom action (apply function)."""
        self.action = action

        if self.is_dynamic and (var is None or types is None):
            raise ValueError("Dynamic rules must have a variablle+types to update.")
        
        self.var = var
        self.types = types




class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name if model_name else "default_model"
        
        self.rules = []
        self.dynamic_rules = []
        self.shooting_rule = None
        self.branching_rule = None
        self.env_reads = {}
        
    def add_rule(self, rule):
        self.rules.append(rule)

    def add_dynamic_rule(self, rule):
        self.dynamic_rules.append(rule)


    def get_trainable_params(self):
        """Return the trainable parameters of the model as one np.array."""
        trainable_params = []
        for rule in self.rules + self.dynamic_rules:
            if rule.trainable:
                trainable_params.append(rule.params)
        return np.concatenate(trainable_params)
    
    def set_trainable_params(self, params):
        """Set the trainable parameters of the model from a np.array, divide based on the rule no_params."""
        start = 0
        for rule in self.rules + self.dynamic_rules:
            if rule.trainable:
                end = start + rule.no_params
                rule.params = params[start:end]
                start = end
                
    def add_shooting_rule(self, rule):
        self.shooting_rule = rule

    def add_branching_rule(self, rule):
        self.branching_rule = rule