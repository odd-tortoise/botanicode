from dataclasses import dataclass, field
from typing import Callable, Any, List, Dict, Optional, Tuple
import numpy as np

class Rule:
    """Base class for all rules."""
    def __init__(self, action: Callable, trainable: bool, no_params: int = 0):
        """
        Initialize a Rule.

        Args:
            action (Callable): The function defining the rule's effect.
            trainable (bool): Whether the rule has trainable parameters.
            no_params (int): Number of trainable parameters.
        """
        self.trainable: bool = trainable
        self.no_params: int = no_params
        self.params: np.ndarray = np.zeros(no_params)
        self.action: Callable = action
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * no_params

    def set_params(self, params: np.ndarray) -> None:
        """
        Set the parameters of the rule.

        Args:
            params (np.ndarray): The parameters to set.
        """
        self.params = params

    def set_bounds(self, bounds: List[Tuple[Optional[float], Optional[float]]]) -> None:
        """
        Set the bounds for the parameters.

        Args:
            bounds (List[Tuple[Optional[float], Optional[float]]]): The bounds to set.
        """
        if len(bounds) != self.no_params:
            raise ValueError(f"Bounds for rule {self} must have the same length as the number of parameters")
        self.bounds = bounds

class StaticRule(Rule):
    """For rules that do not involve time dynamics."""
    def __init__(self, action: Callable, trainable: bool = False, no_params: int = 0):
        """
        Initialize a StaticRule.

        Args:
            action (Callable): The function defining the rule's effect. The callable should have the signature (plant: Plant, params: np.ndarray) -> None.
            trainable (bool): Whether the rule has trainable parameters.
            no_params (int): Number of trainable parameters.
        """
        super().__init__(action, trainable, no_params)

class DynamicRule(Rule):
    """For rules involving time evolution (used in ODE solvers)."""
    def __init__(self, action: Callable, var: str, types: List[type], trainable: bool = False, no_params: int = 0):
        """
        Initialize a DynamicRule.

        Args:
            action (Callable): The function defining the rule's effect.  The callable should have the signature (t: float, y:np.ndarray, plant: Plant, params: np.ndarray) -> np.ndarray.
            var (str): The variable this rule affects.
            types (List[type]): The types of nodes this rule applies to.
            trainable (bool): Whether the rule has trainable parameters.
            no_params (int): Number of trainable parameters.
        """
        super().__init__(action, trainable, no_params)
        self.var: str = var
        self.types: List[type] = types

class DevelopmentEngine:
    """Class representing the development engine."""
    def __init__(self, engine_name: str) -> None:
        """
        Initialize the DevelopmentEngine.

        Args:
            engine_name (str): The name of the engine.
        """
        self.engine_name: str = engine_name if engine_name else "default_engine"
        self.static_rules: List[StaticRule] = []
        self.dynamic_rules: List[DynamicRule] = []
        self.shooting_rule: Optional[Callable] = None
        self.branching_rule: Optional[Callable] = None
        self.env_reads: Dict[str, Any] = {}
        self.loss_functions: List[Callable] = []

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the engine. The order of application of the rules is the order in which they are added.

        Args:
            rule (Rule): The rule to add.
        """
        if isinstance(rule, DynamicRule):
            self.dynamic_rules.append(rule)
        else:
            self.static_rules.append(rule)

    def get_trainable_params(self) -> Optional[np.ndarray]:
        """
        Return the trainable parameters of the model as one np.array.

        Returns:
            Optional[np.ndarray]: The trainable parameters or None if there are no trainable parameters.
        """
        trainable_params = [rule.params for rule in self.static_rules + self.dynamic_rules if rule.trainable]

        return np.concatenate(trainable_params) if trainable_params else None

    def set_trainable_params(self, params: np.ndarray) -> None:
        """
        Set the trainable parameters of the model from a np.array, divide based on the rule no_params. Paramters bounds are enforced.

        Args:
            params (np.ndarray): The parameters to set.
        """
        start = 0
        for rule in self.static_rules + self.dynamic_rules:
            if rule.trainable:
                end = start + rule.no_params
                rule.params = self._apply_bounds(params[start:end], rule.bounds)
                start = end

    def initialize_trainable_params(self) -> None:
        """
        Initialize the trainable parameters of the model.
        """
        trainable_params = self.get_trainable_params()
        if trainable_params is not None:
            #by using the set_trainable_params method, the bounds are enforced
            self.set_trainable_params(np.zeros_like(trainable_params))
            

    def _apply_bounds(self, params: np.ndarray, bounds: List[Tuple[Optional[float], Optional[float]]]) -> np.ndarray:
        """
        Apply bounds to the parameters.

        Args:
            params (np.ndarray): The parameters to apply bounds to.
            bounds (List[Tuple[Optional[float], Optional[float]]]): The bounds to apply.

        Returns:
            np.ndarray: The parameters with bounds applied.
        """
        safe_params = np.empty_like(params)
        for i, (param, (lb, ub)) in enumerate(zip(params, bounds)):
            if lb is not None and ub is not None:
                safe_params[i] = np.clip(param, lb, ub)
            elif lb is not None:
                safe_params[i] = max(param, lb)
            elif ub is not None:
                safe_params[i] = min(param, ub)
            else:
                safe_params[i] = param
        return safe_params

    def add_shooting_rule(self, rule: Callable) -> None:
        """
        Add a shooting rule to the engine.

        Args:
            rule (Callable): The shooting rule to add.  The callable returns a list of tuples (nodes from which to shoot, the amout of shoots to do from that node)

        """
        self.shooting_rule = rule

    def add_branching_rule(self, rule: Callable) -> None:
        """
        Add a branching rule to the engine.

        Args:
            rule (Rule): The branching rule to add.  The callable returns a list of tuples (nodes from which to shoot, the amout of shoots to do from that node)
        """
        self.branching_rule = rule

    def set_env_reads(self, env_reads: Dict[str, Any]) -> None:
        """
        Set the environmental reads for the engine.

        Args:
            env_reads (Dict[str, Any]): The environmental reads to set.
        """
        self.env_reads = env_reads

    def add_loss_function(self, loss_function: Callable) -> None:
        """
        Add a loss function to the engine.

        Args:
            loss_function (Callable): The loss function to add.
        """
        self.loss_functions.append(loss_function)
