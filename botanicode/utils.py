import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

import numpy as np
from scipy.optimize import fsolve  # For solving implicit equations

class NumericalIntegrator:
    def __init__(self, method="forward_euler", dt=0.01):
        """
        Initialize the numerical integrator.

        :param method: Solver method ('forward_euler' or 'backward_euler')
        :param dt: Time step size
        """
        self.method = method
        self.dt = dt
        
    def integrate(self, rhs_function, t, y, rhs_args):
        """
        Perform one integration step using the specified method.

        :param rhs: The RHS function of the ODE, f(t, y)
        :param t: Current time
        :param y: Current state
        :return: New state after one time step
        """
        if self.method == "forward_euler":
            return self.forward_euler(rhs_function,rhs_args, t, y)
        elif self.method == "backward_euler":
            return self.backward_euler(rhs_function,rhs_args, t, y)
        else:
            raise ValueError("Unsupported method: {}".format(self.method))

    def forward_euler(self, rhs_function,rhs_args, t, y):
        """
        Forward Euler method.

        :param rhs: The RHS function of the ODE, f(t, y)
        :param t: Current time
        :param y: Current state
        :return: New state after one time step
        """

        return y + self.dt * rhs_function(t, rhs_args)

    def backward_euler(self, rhs_function,rhs_args, t, y):
        """
        Backward Euler method.

        :param rhs: The RHS function of the ODE, f(t, y)
        :param t: Current time
        :param y: Current state
        :return: New state after one time step
        """
        # Define a function to solve implicitly, FIX
        def implicit_eq(y_next):
            return y_next - y - self.dt * rhs_function(t + self.dt, y_next)

        # Solve the implicit equation using a root-finding method
        y_next = fsolve(implicit_eq, y)  # Use y as the initial guess
        return y_next




# utility functions for plotting and animating the simulation
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
