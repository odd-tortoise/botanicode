import numpy as np
import matplotlib.pyplot as plt

class Hittable:
    def __init__(self, points):
        self.points = points # comprende i punti che definiscono la forma dell'oggetto

class Sky:
    def __init__(self, position):
        self.position = position

        self.range_x = range(-5,5)
        self.range_y = range(-5,5)



    def compute_distance(self, point):
        return np.linalg.norm(self.position - point)
    
    def compute_angle(self, point):
        return np.arctan2(self.position[1] - point[1], self.position[0] - point[0])
    

    def get_plane(self):
        # create x,y
        xx, yy = np.meshgrid(self.range_x, self.range_y)
        normal = self.position / np.linalg.norm(self.position)

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy ) * 1. /normal[2]

        return xx, yy, zz



class Engine:
    def __init__(self, sky):
        self.sky = sky
        self.hittables = []
        self.elevation = 5

    def add_hittable(self, hittable):
        self.hittables.append(hittable)


    def project_hittable(self, hittable):

        def project_point(point):
            return point - np.dot(point, normal) * normal
        
        # project the hittable on the plane defined by the sky position as normal

        # compute the normal to the plane
        normal = self.sky.position / np.linalg.norm(self.sky.position)

    
        # compute the projection of the points on the plane
        projected_points = []
        for point in hittable.points:
            projected_points.append(point - np.dot(point, normal) * normal)

        return np.array(projected_points) + np.array([0,0,self.elevation])


        
    def plot_hittables(self):

        # Plotting in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        xx,yy,zz = self.sky.get_plane()

        ax.plot_surface(xx, yy, zz + self.elevation, alpha=0.1)


        for hittable in self.hittables:
            points = hittable.points
            projected_points = self.project_hittable(hittable)

            ax.scatter(points[:,0], points[:,1], points[:,2], c='r')
            ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], c='b')


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(self.sky.range_x[0], self.sky.range_x[-1])
        ax.set_ylim(self.sky.range_y[0], self.sky.range_y[-1])
        ax.set_zlim(0,10)

        plt.show()


if __name__ == "__main__":

    def generate_circle(radius,points, center, normal):
        angles = np.linspace(0, 2*np.pi, points)
        circle = []
        for theta in angles:
            point = center + radius * np.array([np.cos(theta), np.sin(theta), 0])
            circle.append(point)
        return np.array(circle)
    
    circle1 = generate_circle(1, 100, np.array([0,0,0]), np.array([0,0,1]))
    circle2 = generate_circle(0.5, 100, np.array([1,1,1]), np.array([0,0,1]))

    hittable1 = Hittable(circle1)
    hittable2 = Hittable(circle2)

    sky = Sky(np.array([0,1, 1]))

    engine = Engine(sky)
    engine.add_hittable(hittable1)
    engine.add_hittable(hittable2)

    engine.plot_hittables()




    


