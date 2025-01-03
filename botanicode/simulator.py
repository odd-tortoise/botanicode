import json
import logging


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
        self.total_time = start_time
    
        

    def tick(self, dt):
        """
        Advance the clock by a given time step.
        
        :param dt: Time increment in HOURS.
        """
        # Advance time
        self.elapsed_time += dt
        self.total_time += dt

    def get_hour(self):
        """
        Get the current hour of the day.
        """
        return self.total_time % 24

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

import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation


def plotter(plot_methods, plot_3ds=None, ncols=1, figsize=(10, 10), dpi=100, save_folder=None, name= "plot", save_format='png'):
    """
    Generate the plot and save or show it. Helper function to group multiple plots.
    """
    plot_3ds = plot_3ds if plot_3ds is not None else [False] * len(plot_methods)
    n_objects = len(plot_methods)
    nrows = (n_objects + ncols - 1) // ncols

    # Create subplots with appropriate projections
    fig = plt.figure(figsize=figsize)
    axes = []

    for i, is_3d in enumerate(plot_3ds):
        if is_3d:
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(nrows, ncols, i + 1)
        axes.append(ax)

    # Call the plot methods
    for ax, plot_method in zip(axes, plot_methods):
        plot_method(ax)

    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f"{name}.{save_format}")
        plt.savefig(save_path, dpi=dpi, format=save_format)

    plt.show()

def animate(img_folder, fps=1, save_name="animation.mp4", dpi=100):
    """
    Create an animation from a series of images.
    """
    file_names = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')])

    if not file_names:
        raise ValueError("No images found for animation.")

    # Create a figure and axis for animation
    fig, ax = plt.subplots(dpi=dpi)

    # Function to update the frame
    def update(frame):
        img = plt.imread(file_names[frame], format='png')
        ax.imshow(img)
        ax.axis('off')  # Hide the axes

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(file_names), repeat=False)

    save_path = os.path.join(img_folder, save_name)
    # Save the animation as a video file
    ani.save(save_path, fps=fps, writer='ffmpeg')

    plt.close(fig)

    print(f"Animation completed. Saved to {save_path}.")




class Simulation:

    logging.basicConfig(
    filename="plant_sim.log",
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,

    )
    logger = logging.getLogger("plant_sim")
    
    logger.info("Initializing simulation...")


    @classmethod
    def set_clock(cls, start_time=None, photo_period=(8,18)):
        clock = SimClock(start_time, photo_period)

        cls.clock = clock

        cls.logger.info("Clock loaded.")

    
   
    

    def __init__(self, config_file, env, plant, tasks=None, folder="results"):
        """
        Initialize the Simulation.
        
        :param config: Configuration dictionary or path to configuration file.
        :param clock: SimClock object to use for the simulation.
        """
        
        self.config = self.load_config(config_file)

        self.delta_t = self.config["delta_t"]
        self.steps = self.config["steps"]

        self.env = env
        self.plant = plant

        self.state = {}

        self.before_tasks = tasks["before"] if "before" in tasks else {}
        self.after_tasks = tasks["after"] if "after" in tasks else {}
        self.during_tasks = tasks["during"] if "during" in tasks else {}

        self.folder = folder
        logging.basicConfig(
            filename=folder+"/plant_sim.log",
        )
        

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
            kwargs["name"] = task_name
            if phase == "during" and step is not None:
                task(*args, **kwargs, step=step)
            else:
                task(*args, **kwargs)

    def step(self, delta_t):
        """Advance the simulation by one time step."""
        
        self.plant.grow(Simulation.clock, delta_t, self.env)
        self.env.update(Simulation.clock, delta_t, self.plant)

        self.clock.tick(delta_t)

   
    def run(self, steps, delta_t):
        """Run the simulation for a specified number of steps."""
        if not hasattr(self, 'clock'):
            Simulation.logger.error("A clock must be set up before running the simulation.")
            raise ValueError("A clock must be set up before running the simulation.")
        
        Simulation.logger.info(f"Starting simulation for {steps} steps with delta_t={delta_t}.")

        self.execute_tasks("before")
        
        for step in range(steps):
            Simulation.logger.info(f"Step {step + 1}/{steps}")
            self.step(delta_t)
            self.execute_tasks("during", step=step)

        self.execute_tasks("after")
        self.logger.info("Simulation completed.")


    # fare le cose con lo state, potrebbe essere utile