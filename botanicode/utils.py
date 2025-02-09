import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

import numpy as np
from scipy.optimize import fsolve  # For solving implicit equations
from scipy.integrate import solve_ivp  # For solving ODEs  

class NumericalIntegrator:
    def __init__(self, method="forward_euler"):
        """
        Initialize the numerical integrator.

        :param method: Solver method ('forward_euler' or 'backward_euler')
        :param dt: Time step size
        """
        self.method = method
        self.dt = None

    def set_dt(self, dt):
        """
        Set the time step size.

        :param dt: Time step size
        """
        self.dt = dt
        
    def integrate(self, rhs_function, plant, params, t, y):
        """
        Perform one integration step using the specified method.

        :param rhs: The RHS function of the ODE
        :param t: Current time
        :param y: Current state
        :return: New state after one time step
        """
        if self.method == "forward_euler":
            return self.forward_euler(rhs_function,plant,params, t, y)
        elif self.method == "backward_euler":
            return self.backward_euler(rhs_function,plant,params, t, y)
        else:
            raise ValueError("Unsupported method: {}".format(self.method))

    def forward_euler(self, rhs_function,plant, params, t, y):
        """
        Forward Euler method.

        :param rhs: The RHS function of the ODE
        :param t: Current time
        :param y: Current state
        :return: New state after one time step
        """

        #rhs = rhs_function(t, y, plant, params)

        sol = solve_ivp(rhs_function, [t, t + self.dt], y, args=(plant,params), method='RK45', t_eval = [t + self.dt])
        return sol.y[:,0]
        return y + self.dt * rhs

    def backward_euler(self, rhs_function,plant, params, t, y):
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

from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, simulation, plant, clock, dataset, max_t, delta_t, batch_size=4):
        """
        Run the optimization and return the best parameters along with any additional history/info.
        """
        pass


import numpy as np

class EvolutionaryOptimizer(BaseOptimizer):
    def __init__(self, max_epochs=100, pop_size=20, mutation_scale=0.1, loss_threshold=1e-3):
        self.max_epochs = max_epochs
        self.pop_size = pop_size
        self.mutation_scale = mutation_scale
        self.loss_threshold = loss_threshold

    def optimize(self, simulation, plant, clock, dataset, max_t, delta_t, batch_size=4):
        # Retrieve the current (unconstrained) parameters.
        simulation.model.set_trainable_params(simulation.model.get_trainable_params())
        best_params_unconstrained = simulation.model.get_trainable_params()
        print(f"Initial parameters: {best_params_unconstrained}")   
        
        losses, best_loss = simulation.compute_total_loss(plant, clock, dataset, max_t, delta_t, batch_size)
        history = [(losses, best_loss)]
        print(f"Initial total loss: {best_loss:.4f}")

        for epoch in range(self.max_epochs):
            # Create a population via mutations.
            population = [
                best_params_unconstrained + np.random.randn(*best_params_unconstrained.shape) * self.mutation_scale
                for _ in range(self.pop_size)
            ]
            candidate_losses = []
            candidate_total_loss = []
            candidate_params = []
            for candidate in population:
                simulation.model.set_trainable_params(candidate)
                candidate_params.append(simulation.model.get_trainable_params())
                try:
                    losses_candidate, loss_candidate = simulation.compute_total_loss(plant, clock, dataset, max_t, delta_t, batch_size)
                except Exception as e:
                    print("Error during evaluation:", e)
                    loss_candidate = np.inf
                    losses_candidate = np.array([np.inf] * len(simulation.model.loss_functions))
                candidate_losses.append(losses_candidate)
                candidate_total_loss.append(loss_candidate)

            best_idx = np.argmin(candidate_total_loss)
            if candidate_total_loss[best_idx] < best_loss:
                best_loss = candidate_total_loss[best_idx]
                best_params_unconstrained = candidate_params[best_idx]
                print(f"Epoch {epoch+1}: New best loss = {best_loss:.4f}, params = {best_params_unconstrained}")
            else:
                print(f"Epoch {epoch+1}: No improvement. Best loss remains = {best_loss:.4f}")

            history.append((candidate_losses[best_idx], best_loss))
            # Early stopping
            if best_loss < self.loss_threshold:
                print(f"Early stopping at epoch {epoch+1} with loss {best_loss:.4f}")
                break

            simulation.model.set_trainable_params(best_params_unconstrained)

        # Final transformation before setting the parameters in the model.
        best_parameters = simulation.model.get_trainable_params()
        simulation.model.set_trainable_params(best_parameters)
        return best_parameters, history