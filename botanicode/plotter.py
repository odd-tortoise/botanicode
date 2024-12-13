import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

class Plotter:
    def __init__(self, plot_methods, plot_3ds=None, ncols=1, figsize=(10, 10), dpi=100):
        """
        Args:
            - plot_methods: List of functions or lambdas to be plotted.
            - plot_3ds: List of booleans indicating if the plot is 3D.
            - ncols: Number of columns for the subplot layout.
            - figsize: Size of the figure.
            - dpi: Dots per inch for the figure.
        """
        self.plot_methods = plot_methods
        self.plot_3ds = plot_3ds if plot_3ds is not None else [False] * len(plot_methods)
        self.ncols = ncols
        self.figsize = figsize
        self.dpi = dpi

        
    def plot(self, save_folder=None, name="plot"):
        n_objects = len(self.plot_methods)
        nrows = (n_objects + self.ncols - 1) // self.ncols

        # Create subplots with appropriate projections
        fig = plt.figure(figsize=self.figsize)
        axes = []

        for i, is_3d in enumerate(self.plot_3ds):
            if is_3d:
                ax = fig.add_subplot(nrows, self.ncols, i + 1, projection='3d')
            else:
                ax = fig.add_subplot(nrows, self.ncols, i + 1)
            axes.append(ax)

        # Call the plot methods
        for ax, plot_method in zip(axes, self.plot_methods):
            plot_method(ax)

        plt.tight_layout()

        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, f"{name}.png")
            plt.savefig(save_path, dpi=self.dpi)

        plt.show()

    def animate(self, img_folder, fps=1):
        file_names = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')])

        # Create a figure and axis
        fig, ax = plt.subplots(dpi=self.dpi)

        # Function to update the frame
        def update(frame):
            img = plt.imread(file_names[frame], format='png')
            ax.imshow(img)
            ax.axis('off')  # Hide the axes

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(file_names), repeat=False)

        save_path = os.path.join(img_folder, "animation.mp4")
        # Save the animation as a video file
        ani.save(save_path, fps=fps, writer='ffmpeg')

        plt.close(fig)

        print("Animation completed.")

# Example usage
if __name__ == "__main__":
    def plot_example_2d(ax):
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title("2D Plot")

    def plot_example_3d(ax):
        ax.plot([1, 2, 3], [4, 5, 6], [7, 8, 9])
        ax.set_title("3D Plot")

    plotter = Plotter(plot_methods=[plot_example_2d, plot_example_3d], plot_3ds=[False, True])
    plotter.plot()
