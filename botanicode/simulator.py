import copy
import json
from typing import Optional
from scipy.optimize import minimize

import numpy as np
from botanical_nodes import Stem, Leaf, Root, SAM, RAM

from development_engine import StaticRule, DynamicRule, DevelopmentEngine
from plant import Plant
from env import Environment, Clock
from utils import NumericalIntegrator, BaseOptimizer, EvolutionaryOptimizer

import pickle
import matplotlib.pyplot as plt


class Tracker:
    def __init__(self):
        self.data = {}

        self.data["Plant"] = []
        self.data["Nodes"] = {}

    def snapshot(self, timestamp : float, plant : Plant) -> None:
        """
        Records a snapshot of the state of the system at a given timestamp.

        Args:
            timestamp (float): The current timestamp.
            plant (Plant): The plant instance to snapshot.
        """
        # save the plant data
        self.data["Plant"].append([timestamp,copy.deepcopy(plant.state.__dict__)])

        # save nodes data
        for node in plant.structure.G.nodes():
            node_data_full = copy.deepcopy(node.state.__dict__)
            # keep only the numerical values
            node_data = {key: value for key, value in node_data_full.items() if isinstance(value, (int, float))}
            
            
            node_name = node.name
            node_type = type(node)
            
            if node_type not in self.data["Nodes"]:
                self.data["Nodes"][node_type] = {}
            if node_name not in self.data["Nodes"][node_type]:
                self.data["Nodes"][node_type][node_name] = []

            self.data["Nodes"][node_type][node_name].append([timestamp, node_data])

    def reset(self):
        """
        Resets the tracker data.
        """
        self.data = {}

        self.data["Plant"] = []
        self.data["Nodes"] = {}

        
    def save_to_file(self, path):
        """
        Saves the tracked data to a txt file.

        Args:
            path (str): The file path to save the data.
        """
        try:
            with open(path, 'w') as file:
                # Save plant data
                file.write("Plant\n")
                for timestamp, data in self.data["Plant"]:
                    file.write(f"{timestamp}: {data}\n")

                # Save nodes data
                file.write("\nNodes\n")
                for node_type, values in self.data["Nodes"].items():
                    for node_name, history in values.items():
                        file.write(f"{node_type} - {node_name}\n")
                        for timestamp, data in history:
                            file.write(f"{timestamp}: {data}\n")
                        
            print(f"Data successfully saved to {path}")
        except Exception as e:
            print(f"Error saving data to file: {e}")


    def extract_field_nodes(self, node_type, field):
        """
        Extracts the given field from all snapshots of nodes of a given type.
        
        Args:
            node_type (str): The type of node to extract data from.
            field (str): The field name to extract.
        
        Returns:
            Dict[str, List[Tuple[float, Any]]]: A dictionary where keys are node names and values are lists of tuples (timestamp, field_value).
        """
        result = {}
        if node_type not in self.data["Nodes"]:
            print(f"No data for node type '{node_type}'.")
            return result
        
        for node_name, snapshots in self.data["Nodes"][node_type].items():
            field_values = []
            for ts, state in snapshots:
                # Only add if the field exists in the snapshot
                if field in state:
                    field_values.append((ts, state[field]))
            result[node_name] = field_values
        
        return result
    
    def extract_field_plant(self, field):
        """
        Extracts the given field from all snapshots of the plant.
        
        Args:
            field (str): The field name to extract.
        
        Returns:
            List[Tuple[float, Any]]: A list of tuples (timestamp, field_value).
        """
        result = []
        
        for ts,state in self.data["Plant"]:
            # Only add if the field exists in the snapshot
            if field in state:
                result.append((ts, state[field]))
           
        
        return result

    def extract_field_nodename(self, node_type, node_name, field):
        """
        Extracts the given field from all snapshots of a specific node.

        Args:
            node_type (str): The type of node.
            node_name (str): The name of the node.
            field (str): The field name to extract.

        Returns:
            List[Tuple[float, Any]]: A list of tuples (timestamp, field_value) or an empty list if not found.
        """
        if node_type not in self.data["Nodes"]:
            print(f"No data for node type '{node_type}'.")
            return []
        if node_name not in self.data["Nodes"][node_type]:
            print(f"No data for node '{node_name}' in type '{node_type}'.")
            return []

        field_values = []
        for ts, state in self.data["Nodes"][node_type][node_name]:
            if field in state:
                field_values.append((ts, state[field]))
        return field_values

    def plot_node_field(self, node_type, node_name, field):
        """
        Plots a given field over time for a specific node.

        Args:
            node_type (str): The type of node.
            node_name (str): The name of the node.
            field (str): The field to plot.
           
        """
        # Retrieve the snapshots for the specified node
        snapshots = None
        if node_type in self.data["Nodes"] and node_name in self.data["Nodes"][node_type]:
            snapshots = self.data["Nodes"][node_type][node_name]
        else:
            print(f"No data found for node '{node_name}' of type '{node_type}'.")
            return

        # Extract timestamps and field values
        timestamps = []
        values = []
        for ts, state in snapshots:
            if field in state:
                timestamps.append(ts)
                values.append(state[field])
        
        if not timestamps:
            print(f"Field '{field}' not found in snapshots for node '{node_name}'.")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(timestamps, values, marker='o', linestyle='-')
        plt.xlabel("Time")
        plt.ylabel(field)
        plt.title(f"{node_type} '{node_name}' - {field} over time")
        plt.grid(True)
        plt.show()

    def plot_plant_field(self, field):
        """
        Plots a given field over time for the plant.

        Args:
            field (str): The field to plot.
        """
        snapshots = self.data["Plant"]
        
        # Extract timestamps and field values
        timestamps = []
        values = []
        for ts, state in snapshots:
            if field in state:
                timestamps.append(ts)
                values.append(state[field])
        
        if not timestamps:
            print(f"Field '{field}' not found in snapshots for plant.")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(timestamps, values, marker='o', linestyle='-')
        plt.xlabel("Time")
        plt.ylabel(field)
        plt.title(f"Plant - {field} over time")
        plt.grid(True)
        plt.show()
    
    # TODO: extend these methods to take lists of nodes/types 
    # TODO: make this class more generic to store data also for other purposes (like parameter history)



class Simulation:
    def __init__(self, solver : NumericalIntegrator, folder : str = None):
        """
        Initialize the Simulation.

        Args:
            solver (NumericalIntegrator): The numerical integrator to use for solving differential equations.
            folder (str): The folder to save simulation results.
        """

        self.solver = solver
        self.folder = folder

        self.history = Tracker()

        #self.before_tasks = tasks["before"] if "before" in tasks else {}
        #self.after_tasks = tasks["after"] if "after" in tasks else {}
        #self.during_tasks = tasks["during"] if "during" in tasks else {}

        
    
    def execute_tasks(self, phase, step=None):
       
        """
        Execute tasks for a specific phase, with dynamic and predefined arguments.
        
        :param phase: The phase of execution ("before", "during", or "after").
        :param step: Current simulation step (for "during" tasks).
        """
        if phase == "before":
            tasks = self.before_tasks
        elif phase == "during":
            tasks = self.during_tasks
        elif phase == "after":
            tasks = self.after_tasks
        else:
            raise ValueError("Invalid phase")

        for task_name, task_info in tasks.items():
            task = task_info["method"]
            args = task_info["args"] if "args" in task_info else []
            kwargs = task_info["kwargs"] if "kwargs" in task_info else {}
            
            task(*args, **kwargs)

    
        """Run the simulation for a specified number of steps.
        if not hasattr(self, 'clock'):
            raise ValueError("A clock must be set up before running the simulation.")
        
        Simulation.logger.info(f"Starting simulation with delta_t={self.delta_t}, max_time={self.max_t}")
        step = 0
        self.execute_tasks("before")
        self.plant.snapshot(timestamp = self.clock.get_elapsed_time())
        
        while self.clock.get_elapsed_time() < self.max_t:
            Simulation.logger.info(f"Step {step}: elapsed time={self.clock.get_elapsed_time()}")

            
            # 1) plant read the environment
            self.plant.probe(self.env)

            # 2) solve the system for the whole plant case , one system for each rule
            # this loop can be modified to couple the systems
            
            # assemply the system of differential equations for the plants
            ret_plant = self.plant.get_dynamic_info_plant()

            # solve the system for the whole plant
            for var, values in ret_plant.items(): 
                func = values["ode"]
                nodes = values["node_obj"]
                y = np.array(values["value"])
                t = self.clock.get_elapsed_time()

                new_y = self.solver.integrate(
                    rhs_function = func,
                    t = self.clock.get_elapsed_time(),
                    y = y,
                    rhs_args = (nodes,self.plant)
                    )
                ret_plant[var]["new_value"] = new_y


            # apply the new values to the plant
            self.plant.apply_plant_dynamics(ret_plant)


            
            # 3) solve the system for the nodes , one system for each (node_Type,var) couple
            # this loop can be modified to couple the systems

            #compute the system of differential equations for the nodes 
            ret_nodes = self.plant.get_dynamic_info_nodes()

            for node_type, values in ret_nodes.items():

                nodes = np.array(values["node_obj"])
                func = values["rhs"]
                y = np.array(values["value"])
                t = self.clock.get_elapsed_time()

                new_y = self.solver.integrate(
                    rhs_function = func,
                    rhs_args = nodes,
                    t = t,
                    y = y
                    )
                
                ret_nodes[node_type]["new_value"] = new_y

            # apply the new values to the nodes
            self.plant.apply_node_dynamics(ret_nodes)


            # 4) update the plant and environment
            #update also the derived variables and shoot new parts if needed
            self.plant.grow(self.delta_t,self.env)
            
            # execute the extra tasks for the "during" phase
            self.execute_tasks("during", step=step)

            # Advance the simulation by one time step
            step += 1
            self.clock.tick(self.delta_t)
            self.plant.snapshot(timestamp = self.clock.get_elapsed_time())

        
        self.execute_tasks("after")
        self.logger.info("Simulation completed.")

        self.plant.history.save_to_file(self.folder + "/history.txt")
        


    # fare le cose con lo state, potrebbe essere utile
    """
    
    def run(self, max_t : float, delta_t : float, dev_eng : DevelopmentEngine, plant : Plant, env, clock : Clock):

        """
        Run the simulation for a specified duration.

        Args:
            max_t (float): The maximum simulation time.
            delta_t (float): The time step for the simulation.
            dev_eng (DevelopmentEngine): The development engine instance.
            plant (Plant): The plant instance.
            env (Environment): The environment instance.
            clock (Clock): The clock instance.
        """
        
        self.solver.set_dt(delta_t)
        
        # initialize the trainable parameters to the bounds
        dev_eng.initialize_trainable_params()

        
        self.history.snapshot(timestamp = clock.get_elapsed_time(), plant = plant)


        while(clock.get_elapsed_time() < max_t):
            plant.probe(env, dev_eng.env_reads, clock.get_elapsed_time())

            # apply the rules plant-level
            for rule in dev_eng.dynamic_rules:
                val, nodes = plant.structure.get_nodes_attribute(rule.var, rule.types)

                val = np.array(val)

                new_val = self.solver.integrate(
                    rhs_function = rule.action,
                    t = clock.get_elapsed_time(),
                    y = val,
                    plant = plant,
                    params= rule.params
                    )
                
                
                plant.structure.set_nodes_attribute(rule.var, nodes, new_val)
                    

            for rule in dev_eng.static_rules:
                rule.action(plant, rule.params)

            plant.grow(delta_t,env,dev_eng,clock.get_elapsed_time())
            clock.tick(delta_t)
            self.history.snapshot(timestamp = clock.get_elapsed_time(), plant = plant)

        if self.folder:
            self.history.save_to_file(self.folder + "history.txt")

    @staticmethod
    def compute_loss_for_plant(dev_eng : DevelopmentEngine, data_obs : dict, data_exp : dict):
        """
        Compute the loss for one plant by summing losses from all provided loss functions.

        Args:
            dev_eng (DevelopmentEngine): The development engine instance.
            data_obs (Dict[str, Any]): The observed data.
            data_exp (Dict[str, Any]): The expected data.

        Returns:
            np.ndarray: The computed loss.
        """
        loss = []
        for loss_function in dev_eng.loss_functions:
            loss.append(loss_function(data_obs, data_exp))
        return np.array(loss)

    def compute_total_loss(self, max_t : float , delta_t : float, dataset : list[Environment, dict], batch_size : int, dev_eng : DevelopmentEngine, plant : Plant, clock : Clock):
        """
        Compute the total loss over the dataset.

        Args:
            max_t (float): The maximum simulation time.
            delta_t (float): The time step for the simulation.
            dataset (List[Tuple[Environment, Dict[str, Any]]]): The dataset of environment and target data pairs.
            batch_size (int): The batch size for the simulation.
            dev_eng (DevelopmentEngine): The development engine instance.
            plant (Plant): The plant instance.
            clock (Clock): The clock instance.

        Returns:
            Tuple[np.ndarray, float]: The losses and total loss.
        """
        total_loss = 0.0
        losses = np.array([0.0]*len(dev_eng.loss_functions))

        extracted_samples = np.random.choice(len(dataset), batch_size, replace=False)

        for env, data_target in [dataset[i] for i in extracted_samples]:
            # Create a deep copy of the plant so that each simulation run is independent.
            plant.reset()
            clock.elapsed_time = 0.0
            self.history.reset()
            # Run simulation.
            self.run(max_t, delta_t, dev_eng, plant, env, clock)
            # Compute loss using the dev_eng's loss functions.
            loss = Simulation.compute_loss_for_plant(dev_eng, self.history.data, data_target)
        
            losses+=loss

        losses = losses/batch_size
        
        total_loss = sum(losses)

        return losses, total_loss

    def tune(self, plant : Plant, clock : Clock, dataset : list[Environment, dict], max_t : float, delta_t : float, batch_size : int, dev_eng : DevelopmentEngine, folder : str = None,optimizer : Optional[BaseOptimizer] = None):
        """
        Tune the dev_eng's parameters using the provided optimizer.

        Args:
            plant (Plant): The plant instance.
            clock (Clock): The clock instance.
            dataset (List[Tuple[Environment, Dict[str, Any]]]): The dataset of environment and target data pairs.
            max_t (float): The maximum simulation time.
            delta_t (float): The time step for the simulation.
            batch_size (int): The batch size for the simulation.
            dev_eng (DevelopmentEngine): The development engine instance.
            optimizer (Optional[BaseOptimizer]): An instance of a class that implements BaseOptimizer. If None, a default EvolutionaryOptimizer is used.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: The optimized parameter set (in constrained space) and additional info or history provided by the optimizer.
        """
        if optimizer is None:
            optimizer = EvolutionaryOptimizer()
        best_parameters, optimization_info = optimizer.optimize(simulation=self,dev_eng=dev_eng, plant=plant, clock=clock, dataset=dataset, max_t=max_t, delta_t=delta_t, batch_size=batch_size)

        if folder:
            # Save the optimization info to a file using pickle
            with open(folder + "optimization_info.pkl", "wb") as file:
                pickle.dump(optimization_info, file)
        
            with open(folder + "best_parameters.pkl", "wb") as file:
                pickle.dump(best_parameters, file)


            # plot the best loss
            plt.semilogy([x[1] for x in optimization_info])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training loss")
            plt.savefig(folder+"loss.png")

            # plot the losses
            plt.semilogy([x[0] for x in optimization_info])
            plt.xlabel("Epoch")
            plt.ylabel("Losses")
            plt.title("Training losses")
            plt.savefig(folder+"losses.png")

        return best_parameters, optimization_info