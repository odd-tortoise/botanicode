import json

import numpy as np

class SimClock:
    def __init__(self, start_time=None, photo_period=(8,18)):
        """
        Initialize the simulation clock.
        
        :param start_time: Initial time in hours (random if None).
        :param photo_period: Length of the daylight period in hours.
        """
        # Start time is random within a 24-hour day if not provided
        self.elapsed_time = 0
        self.start_time = start_time if start_time is not None else 0
        self.photo_period = photo_period
        
    
        

    def tick(self, dt):
        """
        Advance the clock by a given time step.
        
        :param dt: Time increment in HOURS.
        """
        # Advance time
        self.elapsed_time += dt
        

    def get_hour(self):
        """
        Get the current hour of the day.
        """
        return (self.start_time+ self.elapsed_time) % 24

    def is_day(self):
        """
        Check if it's currently day based on the photo period.
        """
        return self.get_hour() >= self.photo_period[0] and self.get_hour() <= self.photo_period[1] 


    def get_elapsed_time(self):
        """
        Get the elapsed_time time in hours since the simulation started.
        """
        return self.elapsed_time
    

    def summary(self):
        """
        Provide a summary of the clock's state.
        """
        return {
            "Elapsed Time (hrs)": self.elapsed_time,
            "Current Hour": self.get_hour(),
            "Is Day": self.is_day(),
        }

class Simulation:

    @classmethod
    def set_clock(cls, start_time=None, photo_period=(8,18)):
        clock = SimClock(start_time, photo_period)
        cls.clock = clock
 
    def __init__(self, config_file, env, plant, solver, model, tasks=None, folder="results"):
        """
        Initialize the Simulation.
        
        :param config: Configuration dictionary or path to configuration file.
        :param clock: SimClock object to use for the simulation.
        """
        
        self.config = self.load_config(config_file)
        self.max_t = self.config["max_t"]
        self.delta_t = solver.dt
       
        self.solver = solver
        self.model = model

        self.env = env
        self.plant = plant

        # set the clock for the plant and the environment
        self.plant.set_clock(Simulation.clock)
        self.env.set_clock(Simulation.clock)

        self.before_tasks = tasks["before"] if "before" in tasks else {}
        self.after_tasks = tasks["after"] if "after" in tasks else {}
        self.during_tasks = tasks["during"] if "during" in tasks else {}

        self.folder = folder

        
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

    def run(self):
        """Run the simulation for a specified number of steps."""
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

            # 2) compute the system of differential equations
            ret = self.plant.get_dynamic_info()
            
            # 3) solve the system
            for node_type, values in ret.items():
                nodes = np.array(values["node_obj"])
                rhs = values["rhs"]
                y = np.array(values["value"])
                new_y = self.solver.integrate(
                    rhs_function = rhs,
                    t = self.clock.get_elapsed_time(),
                    y = y,
                    rhs_args = nodes)
                
                ret[node_type]["new_value"] = new_y


            # 4) update the plant and environment
            # use the solution of the system to update the plant, update also the derived variables
            self.plant.grow(ret,self.delta_t)

            # we need to probe the environment again to give the new values to the plant new parts
            self.plant.probe(self.env)

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