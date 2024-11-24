import matplotlib.pyplot as plt

import matplotlib.animation as animation
import os

class Plotter:
    def __init__(self, objects_to_plot, plot_methods, plot_3ds, ncols=2, figsize=(12, 6), dpi = 400):
        """
        A class to handle the plotting of multiple objects with `plot` methods.

        Args:
        - objects_to_plot: List of objects with their `plot` methods to be plotted.
        - ncols: Number of columns for the subplot layout.
        - figsize: Size of the figure.
        """
        self.objects_to_plot = objects_to_plot
        self.plot_methods = plot_methods if plot_methods is not None else [None] * len(objects_to_plot)
        self.plot_3d = plot_3ds

        if len(objects_to_plot) != len(plot_methods) or len(objects_to_plot) != len(plot_3ds):
            raise ValueError("The number of objects and plot methods must be the same.")
        self.ncols = ncols
        self.figsize = figsize
        self.dpi = dpi

        self.id = 0

    def add_object(self, obj, plot_method = None, plot_3d = False):
        """
        Method to add an object to the list of objects to plot.
        """
        self.objects_to_plot.append(obj)
        self.plot_methods.append(plot_method)
        self.plot_3d.append(plot_3d)

    def plot(self, save_folder=None, name="plot"):
        n_objects = len(self.objects_to_plot)
        nrows = (n_objects + self.ncols - 1) // self.ncols

        

        # Create subplots with appropriate projections
        fig = plt.figure(figsize=self.figsize)
        axes = []

        for i,plottype in enumerate(self.plot_3d):
            if plottype:
                ax = fig.add_subplot(nrows, self.ncols, i + 1, projection='3d')
            else:
                ax = fig.add_subplot(nrows, self.ncols, i + 1)
            axes.append(ax)


        # Plot each object using the corresponding axis
        for i, zipped in enumerate(zip(self.objects_to_plot, self.plot_methods)):
            ax = axes[i]

            obj, method = zipped

            if method is not None:
                plot_method = getattr(obj, method)
                plot_method(ax=ax)
            else:
                obj.plot(ax=ax)
            
            ax.set_title(f"{obj.__class__.__name__}")

        
        plt.tight_layout()
        
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # format the id number to 3 digits
           
            id = self.id
            idfilled = str(id).zfill(3)
            
            name = f"{name}_{idfilled}.png"

            full_save_path = os.path.join(save_folder, name)
            plt.savefig(full_save_path, dpi=self.dpi)
            self.id += 1
        else:
            plt.show()

        plt.close(fig)

    def animate(self, img_folder, fps=2):
        """
        Method to create an animation of the objects to plot.
        """
        
        # Get the list of image files in the folder

        print("Working on the animation...")

        file_names = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')]

        # sort the file names
        file_names.sort()


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
        ani.save(save_path,fps=fps, writer='ffmpeg')

        plt.close(fig)

        print("Animation completed.")
