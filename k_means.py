import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Centroid:
    """
    pos    = [x, y] coordinate array
    points = points assigned to centroid
    """
    def __init__(self, pos):
        self.pos = pos
        self.points = []
        self.color = None


class KMeans:
    """
    Unsupervised clustering algortihm.
    """
    def __init__(self, n_centroids):
        self.n_centroids = n_centroids

        self.centroids = []

        # generate initial centroids
        r = lambda: np.random.randint(1, 100)
        for _ in range(n_centroids):
            self.centroids.append(Centroid(np.array([r(), r()])))
        
        # assign a color to each centroid
        colors = cm.rainbow(np.linspace(0, 1, len(self.centroids)))
        for i, c in enumerate(self.centroids):
            c.color = colors[i]


    def fit(self, X, epochs=10):
        """
        Assigns points to centroids.
        Calls to update centroid mean to reflect mean of assigned points.
        """
        self.X = X
        for epoch in range(epochs):
            for point in X:
                distances = {}
                for centroid in self.centroids:
                    distances[centroid] = self._euclidean_distance(centroid.pos, point)
                closest = min(distances.items(), key=operator.itemgetter(1))[0]
                closest.points.append(point)

            self._update_centroids() if epoch != epochs - 1 else self._update_centroids(reset=False)
    
    def _update_centroids(self, reset=True):
        """
        Updates centroid position based on mean of assigned points.
        """
        for centroid in self.centroids:
            x_cor = [x[0] for x in centroid.points]
            y_cor = [y[0] for y in centroid.points]
            try:
                centroid.pos[0] = sum(x_cor)/len(x_cor)
                centroid.pos[1] = sum(y_cor)/len(y_cor)
            except:
                pass

            if reset:
                centroid.points = []
        
    def _euclidean_distance(self, a, b):
        """
        Returns euclidean distance between two points.
        """
        dist = np.linalg.norm(a-b)
        return dist

    def show(self):
        """
        Displays clustering.
        """

        for i, c in enumerate(self.centroids):
            plt.scatter(c.pos[0], c.pos[1], marker='o', color=c.color, s=75)
            x_cors = [x[0] for x in c.points]
            y_cors = [y[1] for y in c.points]
            plt.scatter(x_cors, y_cors, marker='.', color=c.color)

        title = 'k_means'
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        
        plt.savefig('{}.png'.format(title))

        plt.show()


if __name__ == '__main__':

    # sample data
    r = lambda: np.random.randint(1, 100)
    X = [[r(), r()] for _ in range(100)]

    # K-Means instance
    kmeans = KMeans(4)
    kmeans.fit(X, epochs=5)
    kmeans.show()

