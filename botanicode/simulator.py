import json
from scipy.optimize import minimize

import numpy as np
from botanical_nodes import Stem, Leaf, Root, SAM, RAM
from env import Environment
from utils import NumericalIntegrator

class SimClock:
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
 
    def __init__(self, plant, model, clock : SimClock, env: Environment, solver : NumericalIntegrator):
        """
        Initialize the Simulation.
        
        :param config: Configuration dictionary or path to configuration file.
        :param clock: SimClock object to use for the simulation.
        """
        
        
        self.model = model
        self.env = env
        self.plant = plant
        self.clock = clock
        self.solver = solver



        #self.before_tasks = tasks["before"] if "before" in tasks else {}
        #self.after_tasks = tasks["after"] if "after" in tasks else {}
        #self.during_tasks = tasks["during"] if "during" in tasks else {}

        #self.folder = folder

        
    def load_config(self, config):
        """Load configuration from a file or dictionary."""
        if isinstance(config, str):  # File path
            with open(config, 'r') as f:
                return json.load(f)
        elif isinstance(config, dict):
            return config
        else:
            raise ValueError("Config must be a file path or a dictionary.")
        
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

    def run(self, max_t, delta_t):
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
        import matplotlib.pyplot as plt
        self.plant.snapshot(timestamp = self.clock.get_elapsed_time())
        while(self.clock.elapsed_time < max_t):
            self.plant.probe(self.env, self.model.env_reads,self.clock.elapsed_time)

            # apply the rules plant-level
            for rule in self.model.dynamic_rules:
                self.plant.apply_dynamic_rule(rule,self.clock.elapsed_time,self.solver)

            for rule in self.model.rules:
                self.plant.apply_rule(rule)

            self.plant.grow(delta_t,self.env,self.model,self.clock.elapsed_time)
            self.clock.tick(delta_t)
            self.plant.snapshot(timestamp = self.clock.get_elapsed_time())


        

    def reset_simulation(self):
        """
        Reset the simulation to the initial state.
        """
        self.plant.reset()
        self.clock.elapsed_time = 0


    def tune(self, dataset, max_t, delta_t):
        """
        Tune the parameters of the model using the provided dataset.
        """
        # Extract trainable parameters from all rules

        import matplotlib.pyplot as plt
        initial_params = self.model.get_trainable_params()

        def loss_fn(params):
            """
            Loss function for optimization.
            """
            
            self.model.set_trainable_params(params)  # Update parameters
            total_loss = 0

            # Evaluate the model
            # questo dorebbe essre un loop su tutti i dati-env
            self.reset_simulation()  # Reset plant to initial state
            self.run(max_t=max_t, delta_t=delta_t)

            for key, value in dataset.data.items():
                # key is the internode number
                # value is the list of lengths vs time
                target = value
                timestamps,simulated_lengths = self.plant.history.get_variable_over_time("Stem","S"+str(key+1),"length")
                
                # compute the loss considering the lengths of the internodes at the same time
                # pad the simulated lengths with zeros

                total_loss += np.sum((np.array(target) - np.array(simulated_lengths))**2)


            print(total_loss)
            return total_loss

        # Optimize parameters (e.g., using gradient descent or Scipy optimization)
        result = minimize(loss_fn, initial_params, method="L-BFGS-B")
        self.model.set_trainable_params(result.x)


