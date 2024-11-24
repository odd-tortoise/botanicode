import numpy as np
import matplotlib.pyplot as plt

class LevelSetMethod:
    def __init__(self, x,y,nx, ny, dt, vector_field, phi_initial_func, 
                 spatial_scheme="upwind"):
        """
        Initialize the level set method solver.

        Parameters:
        - phi_initial: Initial level set function (2D numpy array).
        - dx, dy: Grid spacing in x and y directions.
        - spatial_scheme: 'upwind' or 'eno' for spatial discretization.
        """

        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.vector_field = vector_field
        
        # Create spatial grid
        self.x = x
        self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Initialize phi
        self.phi = phi_initial_func(self.X, self.Y)

        self.spatial_scheme = spatial_scheme.lower()
        if self.spatial_scheme not in ['upwind', 'eno']:
            raise ValueError("spatial_scheme must be 'upwind' or 'eno'")

    def compute_grad_phi(self):
        """
        Compute spatial gradients of phi using the selected spatial scheme.

        Parameters:
        - F_x, F_y: Velocity fields in x and y directions.

        Returns:
        - phi_x: Approximate gradient of phi in the x-direction.
        - phi_y: Approximate gradient of phi in the y-direction.
        """
        if self.spatial_scheme == 'upwind':
            phi_x, phi_y = self.upwind_gradient()
        elif self.spatial_scheme == 'eno':
            phi_x = self.eno_gradient(axis=1)
            phi_y = self.eno_gradient(axis=0)
        return phi_x, phi_y

    def upwind_gradient(self):
        """
        Compute upwind gradients of phi.

        Parameters:
        - F_x, F_y: Velocity fields in x and y directions.

        Returns:
        - phi_x: Upwind gradient in the x-direction.
        - phi_y: Upwind gradient in the y-direction.
        """
        phi = self.phi
        F_x, F_y = self.vector_field.get_field()

        # Forward and backward differences in x
        phi_x_forward = (np.roll(phi, -1, axis=1) - phi) / self.dx
        phi_x_backward = (phi - np.roll(phi, 1, axis=1)) / self.dx

        # Forward and backward differences in y
        phi_y_forward = (np.roll(phi, -1, axis=0) - phi) / self.dy
        phi_y_backward = (phi - np.roll(phi, 1, axis=0)) / self.dy

        # Select upwind difference based on sign of velocity
        phi_x = np.where(F_x >= 0, phi_x_backward, phi_x_forward)
        phi_y = np.where(F_y >= 0, phi_y_backward, phi_y_forward)

        return phi_x, phi_y

    def eno_gradient(self,axis):
        """
        Compute gradient using a first-order ENO scheme.

        Parameters:
        - phi: Level set function.
        - axis: Axis along which to compute the gradient (0 for y, 1 for x).

        Returns:
        - grad_phi: Approximate gradient of phi along the specified axis.
        """
        phi = self.phi

        shift_minus = np.roll(phi, 1, axis=axis)
        shift_plus = np.roll(phi, -1, axis=axis)
        shift_minus2 = np.roll(phi, 2, axis=axis)
        shift_plus2 = np.roll(phi, -2, axis=axis)

        dx = self.dx if axis == 1 else self.dy

        # First-order differences
        D0_minus = (phi - shift_minus) / dx
        D0_plus = (shift_plus - phi) / dx

        # Second-order differences
        D1_minus = (shift_minus - shift_minus2) / dx
        D1_plus = (shift_plus2 - shift_plus) / dx

        # Smoothness indicators
        S_minus = np.abs(D0_minus - D1_minus)
        S_plus = np.abs(D0_plus - D1_plus)

        ENO_condition = S_minus <= S_plus

        grad_phi = np.where(ENO_condition, D0_minus, D0_plus)
        return grad_phi

    def compute_time_step(self,phi_x, phi_y, dt):
        """
        Update the level set function phi.

        Parameters:
        - F_x, F_y: Velocity fields in x and y directions.
        - phi_x, phi_y: Spatial gradients of phi.
        - dt: Time step size.
        """
        # Update phi using Forward Euler method
        F_x, F_y = self.vector_field.get_field()
        self.phi -= dt * (F_x * phi_x + F_y * phi_y)

    def time_step(self):
        """
        Perform one time step.

        Parameters:
        - F_x, F_y: Velocity fields in x and y directions.
        - dt: Time step size.
        """
        # Compute spatial gradients
        phi_x, phi_y = self.compute_grad_phi()

        # Update phi
        self.compute_time_step(phi_x, phi_y, self.dt)

        self.vector_field.compute_field(self.phi, (phi_x, phi_y))   

        # CFL condition print
        max_magnitude = self.vector_field.get_max_magnitude()
        print(f"Max magnitude: {max_magnitude:.2f}")

        dt_max = min(self.dx, self.dy) / max_magnitude
        print(f"Max time step: {dt_max:.4f}")

    def reinitialize(self, iterations=10):
        """
        Reinitialize phi to be a signed distance function.

        Parameters:
        - iterations: Number of iterations for reinitialization.
        """
        phi = self.phi.copy()
        sign_phi = phi / np.sqrt(phi**2 + 1e-12)
        dtau = 0.5 * min(self.dx, self.dy)

        for _ in range(iterations):
            phi_x_forward = (np.roll(phi, -1, axis=1) - phi) / self.dx
            phi_x_backward = (phi - np.roll(phi, 1, axis=1)) / self.dx
            phi_y_forward = (np.roll(phi, -1, axis=0) - phi) / self.dy
            phi_y_backward = (phi - np.roll(phi, 1, axis=0)) / self.dy

            # Godunov scheme for reinitialization
            phi_x_positive = np.maximum(phi_x_backward, 0)
            phi_x_negative = np.minimum(phi_x_forward, 0)
            phi_y_positive = np.maximum(phi_y_backward, 0)
            phi_y_negative = np.minimum(phi_y_forward, 0)

            grad_phi_positive = np.sqrt(phi_x_positive**2 + phi_y_positive**2)
            grad_phi_negative = np.sqrt(phi_x_negative**2 + phi_y_negative**2)

            grad_phi = np.where(phi > 0, grad_phi_positive, grad_phi_negative)

            phi -= dtau * (grad_phi - 1) * sign_phi

        self.phi = phi

    def get_phi(self):

        """
        Return the current level set function.

        Returns:
        - phi: Current level set function.
        """
        return self.phi.copy()
    

    def plot(self, ax=None):
        """
        Plot the level set function phi.
        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.contour(self.phi, levels=[0], colors="red", linewidths=2, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        ax.imshow(self.phi, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin="lower", cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        

        # limit the plot to the domain
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        # Add a colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax)
        cbar.set_label("Phi")

        if ax is None:
            plt.show()
        


    def is_3d_plot(self):
        return False