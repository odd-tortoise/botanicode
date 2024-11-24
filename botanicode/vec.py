import numpy as np
import matplotlib.pyplot as plt


class Tropism:
    def compute_field(self, X, Y, phi, grad_phi):
        """Compute the tropism vector field (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement `compute_field`")
        
class Gravitropism(Tropism):
    def compute_field(self, X, Y, phi, grad_phi):

        F_x = np.zeros_like(phi)
        F_y = 9.81 * np.ones_like(phi)  # Apply the weight to modulate the strength

        return F_x, F_y

class Phototropism(Tropism):
    def __init__(self, Sky):
        super().__init__()
        self.sky = Sky

    def compute_field(self, X, Y, phi, grad_phi):
        
        # Compute the light intensity at each point of the grid

        x = np.zeros_like(phi)
        y = np.zeros_like(phi)
        distance = np.zeros_like(phi)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                
                x[i,j] = self.sky.position[0] - X[i,j]
                y[i,j] = self.sky.position[1] - Y[i,j]

                # Compute the distance between the point and the light source
                distance[i,j] = np.sqrt(( self.sky.position[0] - X[i,j])**2 + ( self.sky.position[1] - Y[i,j])**2)

        F_x = x*distance
        F_y = y*distance
        return F_x, F_y


class VectorField:
    def __init__(self, x,y,nx,ny):
        self.x = x  
        self.y = y
        self.nx = nx
        self.ny = ny
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.tropisms = []    # List of tropism objects
        self.weights = []     # Weights for each tropism
        self.phi = None
        self.grad_phi = None
        self.F_x = None       # Combined vector field components
        self.F_y = None

        self.compute_field(self.X, self.Y)


    def scale_field(self):

        base_y = np.min(self.Y[self.phi < 0])
        top_y = np.max(self.Y[self.phi < 0])

        
        # compute weights for scaling the field
        weight = np.zeros_like(self.phi)
       
        # weights ramp from zero to one from base_y to top_y
        weight = (self.Y - base_y) / (top_y - base_y)
        weight[weight < 0] = 0
        weight[weight > 1] = 1

        # scale the field
        self.F_x *= weight
        self.F_y *= weight



    def add_tropism(self, tropism, weight=1.0):
        """Add a tropism with a specified weight."""
        self.tropisms.append(tropism)
        self.weights.append(weight)
        

    def normalize_weights(self):
        """Normalize weights to ensure they sum to 1."""
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        print("Tropism weights:", self.weights)

    def compute_field(self, phi, grad_phi):
        """Compute the combined vector field from all tropisms."""
        self.normalize_weights()
        self.phi = phi
        self.grad_phi = grad_phi
        F_x_total = np.zeros_like(self.phi)
        F_y_total = np.zeros_like(self.phi)
        for tropism, weight in zip(self.tropisms, self.weights):
            F_x, F_y = tropism.compute_field(self.X, self.Y, self.phi, self.grad_phi)
            F_x_total += weight * F_x
            F_y_total += weight * F_y
        self.F_x = F_x_total
        self.F_y = F_y_total

        self.scale_field()
    
    def get_field(self):
        """Return the computed vector field components."""
        return self.F_x, self.F_y
    
    def get_max_magnitude(self):
        """Return the maximum magnitude of the vector field."""
        return np.sqrt(self.F_x**2 + self.F_y**2).max()

    def is_3d_plot(self):
        return False

    def plot(self, ax=None, stride=20):
        """
        Plot the vector field using a subset of the space data to avoid overcrowding.
        
        Parameters:
        - ax: matplotlib Axes object (optional)
        - stride: int, step size for slicing the data to reduce the number of vectors plotted
        """
        if self.F_x is None or self.F_y is None:
            raise ValueError("Vector field not computed. Call `compute` method first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            

        # Subset the data
        X_sub = self.X[::stride, ::stride]
        Y_sub = self.Y[::stride, ::stride]
        F_x_sub = self.F_x[::stride, ::stride]
        F_y_sub = self.F_y[::stride, ::stride]

        # Plot the vector field
        ax.quiver(X_sub, Y_sub, F_x_sub, F_y_sub, color='blue', scale=50)
        ax.streamplot(self.X, self.Y, self.F_x, self.F_y, color='black', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

           # limit the plot to the domain
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])
       