import json
from scipy.optimize import minimize

import numpy as np
from botanical_nodes import Stem, Leaf, Root, SAM, RAM

from model import Model
from env import Environment
from utils import NumericalIntegrator, BaseOptimizer, EvolutionaryOptimizer

class Clock:
    def __init__(self, photo_period=(8, 18), step="hour"):
        """
        Initialize the simulation clock.
        :param photo_period: Tuple defining the start and end of the daylight period in hours.
        :param step: Simulation step mode, either "hour" or "day".
        """
        self.elapsed_time = 0  # Total elapsed time in hours always
        self.photo_period = photo_period
        self.step = step  # Step mode: "hour" or "day"

    def tick(self, dt):
        """
        Advance the clock by a given time step.

        :param dt: Time increment (in hours if step="hour", or days if step="day").
        """
        if self.step == "hour":
            self.elapsed_time += dt
        elif self.step == "day":
            self.elapsed_time += dt * 24
        else:
            raise ValueError("Unsupported step mode. Use 'hour' or 'day'.")

    def get_hour(self):
        """
        Get the current hour of the day.
        :return: Integer representing the current hour (0-23).
        """
        if self.step == "hour":
            return int(self.elapsed_time % 24)
        else:
            raise ValueError("Use hour step mode to get info on the current hour.")

    def get_day(self):
        """
        Get the current day of the simulation.
        :return: Integer representing the current day.
        """
        if self.step in ["hour", "day"]:
            return int(self.elapsed_time // 24)
        else:
            raise ValueError("Unsupported step mode. Use 'hour' or 'day'.")

    def is_day(self):
        """
        Check if it's currently daytime based on the photo period.
        :return: Boolean indicating if it's currently day.
        """
        if self.step == "hour":
            current_hour = self.get_hour()
            return self.photo_period[0] <= current_hour < self.photo_period[1]
        else: 
            raise ValueError("Use hour step mode to get info on day/night.")

    def get_elapsed_time(self):
        """
        Get the elapsed time in hours since the simulation started.
        :return: Float representing the elapsed time in hours.
        """
        if self.step == "hour":
            return self.elapsed_time
        elif self.step == "day":
            return self.elapsed_time / 24

    def summary(self):
        """
        Provide a summary of the clock's state.
        :return: Dictionary summarizing the simulation clock's state.
        """
        return {
            "Elapsed Time (hrs)": self.elapsed_time
        }

class Simulation:
    def __init__(self, solver : NumericalIntegrator, folder : str, model : Model):
        """
        Initialize the Simulation.
        
        :param config: Configuration dictionary or path to configuration file.
        :param clock: SimClock object to use for the simulation.
        """

        self.solver = solver

        #self.before_tasks = tasks["before"] if "before" in tasks else {}
        #self.after_tasks = tasks["after"] if "after" in tasks else {}
        #self.during_tasks = tasks["during"] if "during" in tasks else {}

        self.folder = folder
        self.model = model

        
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
    
    def run(self, max_t, delta_t, plant, env, clock):
        self.solver.set_dt(delta_t)
        
        # clip the trainable parameters to the bounds
        if self.model.get_trainable_params() is not None:
            self.model.set_trainable_params(self.model.get_trainable_params())
        
        plant.snapshot(timestamp = clock.get_elapsed_time())
        while(clock.elapsed_time < max_t):
            plant.probe(env, self.model.env_reads,clock.elapsed_time)

            # apply the rules plant-level
            for rule in self.model.dynamic_rules:
                plant.apply_dynamic_rule(rule,clock.elapsed_time,self.solver)

            for rule in self.model.static_rules:
                rule.apply(plant)

            plant.grow(delta_t,env,self.model,clock.elapsed_time)
            clock.tick(delta_t)
            plant.snapshot(timestamp = clock.get_elapsed_time())

        #plant.history.save_to_file(self.folder + "history.txt")

    @staticmethod
    def compute_loss_for_plant(model, history_obs, history_exp):
        """
        Compute the loss for one plant by summing losses from all provided loss functions.
        We assume that model.loss_functions is a list of callable loss functions.
        Each loss function takes (plant_obs, plant_exp) as arguments.
        """
        loss = []
        for loss_function in model.loss_functions:
            loss.append(loss_function(history_obs, history_exp))
        return np.array(loss)

    def compute_total_loss(self, plant, clock, dataset, max_t, delta_t, batch_size):
        """
        Compute the total loss over the dataset.
        For each (env, plant_target) pair in the dataset, run the simulation and compute the loss.
        
        """
        total_loss = 0.0
        losses = np.array([0.0]*len(self.model.loss_functions))
        num_samples = batch_size
        extracted_samples = np.random.choice(len(dataset), num_samples, replace=False)


        for env, hist_target in [dataset[i] for i in extracted_samples]:
            # Create a deep copy of the plant so that each simulation run is independent.
            plant.reset()
            clock.elapsed_time = 0
            # Run simulation.
            self.run(max_t, delta_t, plant, env, clock)
            # Compute loss using the model's loss functions.
            loss = Simulation.compute_loss_for_plant(self.model, plant.history.data, hist_target.data)
        
            losses+=loss

        losses = losses/num_samples
        
        total_loss = sum(losses)

        return losses, total_loss

    def tune(self, plant, clock, dataset, max_t, delta_t, batch_size=4, optimizer: BaseOptimizer = None):
        """
        Tune the model's parameters using the provided optimizer.
        
        Parameters:
            plant, clock, dataset, max_t, delta_t, batch_size: Simulation parameters.
            optimizer: An instance of a class that implements BaseOptimizer.
                       If None, a default EvolutionaryOptimizer is used.
        
        Returns:
            best_parameters: The optimized parameter set (in constrained space).
            optimization_info: Additional info or history provided by the optimizer.
        """
        if optimizer is None:
            optimizer = EvolutionaryOptimizer()
        best_parameters, optimization_info = optimizer.optimize(self, plant, clock, dataset, max_t, delta_t, batch_size)
        return best_parameters, optimization_info