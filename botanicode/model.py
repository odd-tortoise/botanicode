from dataclasses import dataclass, field
from typing import Dict, Callable, Any, List, Tuple
import numpy as np


from typing import List, Callable
import numpy as np

import numpy as np
from typing import Callable, List, Optional, Dict

class Rule:
    """Base class for all rules."""
    def __init__(self, action: Callable, trainable: bool, no_params: int = 0):
        self.trainable = trainable
        self.no_params = no_params
        self.params = np.zeros(no_params)
        self.action = action  # The function defining the rule's effect
        self.bounds = [(None, None)]*no_params  # Default bounds for all parameters

    def set_params(self, params):
        self.params = params

    def set_bounds(self, bounds):
        if len(bounds) != self.no_params:
            raise ValueError(f"Bounds for rule {self} must have the same length as the number of parameters")
        self.bounds = bounds

class StaticRule(Rule):
    """For rules that do not involve time dynamics."""
    def __init__(self, action: Callable, trainable: bool = False, no_params: int = 0):
        super().__init__(action, trainable, no_params)

    def apply(self, plant):
        return self.action(plant, self.params)  # Apply with trainable parameters
      
class DynamicRule(Rule):
    """For rules involving time evolution (used in ODE solvers)."""
    def __init__(self, action: Callable, var: str, types: List[type], trainable: bool = False, no_params: int = 0):
        super().__init__(action, trainable, no_params)
        self.var = var  # The variable this rule affects
        self.types = types  # The types of nodes this rule applies to

# StaticRule (for direct state updates) -> Calls its action(plant, params).
# DynamicRule (for time-based evolution) -> Calls its rhs(t, y, plant, params).


class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name if model_name else "default_model"
        
        self.static_rules = []
        self.dynamic_rules = []
        self.shooting_rule = None
        self.branching_rule = None
        self.env_reads = {}
        self.loss_functions = []
       
        
    def add_rule(self, rule):
        if isinstance(rule, DynamicRule):
            self.dynamic_rules.append(rule)
        else:
            self.static_rules.append(rule)


    def get_trainable_params(self):
        """Return the trainable parameters of the model as one np.array."""
        trainable_params = []
        for rule in self.static_rules + self.dynamic_rules:
            if rule.trainable:
                trainable_params.append(rule.params)
        
        if len(trainable_params) == 0:
            return None
        return np.concatenate(trainable_params)
    
    def set_trainable_params(self, params):
        """Set the trainable parameters of the model from a np.array, divide based on the rule no_params."""
        start = 0
        for rule in self.static_rules + self.dynamic_rules:
            if rule.trainable:
                safe_params = np.empty_like(rule.params)
                end = start + rule.no_params
                params_unconstrained = params[start:end]

                for i, (lb, ub) in enumerate(rule.bounds):
                    if lb is not None and ub is not None:
                        safe_params[i] = np.clip(params_unconstrained[i], lb, ub)
                    elif lb is not None:
                        safe_params[i] = max(params_unconstrained[i], lb)
                    elif ub is not None:
                        safe_params[i] = min(params_unconstrained[i], ub)
                    else:
                        safe_params[i] = params_unconstrained[i]
                
                rule.params = safe_params
                start = end

    
    def add_shooting_rule(self, rule):
        self.shooting_rule = rule

    def add_branching_rule(self, rule):
        self.branching_rule = rule

    def add_loss_function(self, loss_function):
        self.loss_functions.append(loss_function)
    
