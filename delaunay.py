import numpy as np
import random
import networkx as nx

class Graph:

    def __init__(self, n):
        self.n = n
        self.m = 0

        self.edges = [[] for i in range(n)]
        self.weight = np.full((n, n), np.inf)

        self.s = 0;
        self.t = 0;

    def getTitle(self):
        return self.title

    def addEdge(self, i, j, w):
        self.edges[i].append(j)
        self.edges[j].append(i)

        self.weight[i, j] = w
        self.weight[j, i] = w
        
        self.m += 1

    def removeEdge(self, i, j):
        if i in self.edges[j]:
            self.edges[i].remove(j)
            self.edges[j].remove(i)

            self.weight[i, j] = np.inf
            self.weight[j, i] = np.inf

            self.m -= 1

    def setSource(self, s):
        self.s = s

    def setSink(self, t):
        self.t = t

    def shortestPath(self, s = None, t = None):
        marqued = np.full(self.n, True, dtype=bool)
        value = np.full(self.n, np.inf)
        pred = np.full(self.n, -1, dtype=int)

        if(s == None):
            s = self.s
        if(t == None):
            t = self.t

        value[s] = 0
        while(value[t] == np.inf):
            k = -1
            mi = np.inf
            for i in range(self.n):
                if(marqued[i] and mi > value[i]):
                    k = i
                    mi = value[i]
            for j in self.edges[k]:
                if(marqued[j] and value[j] > value[k] + self.weight[k][j]):
                    value[j] = value[k] + self.weight[k][j]
                    pred[j] = k

            marqued[k] = False

        path = [t]
        a = t
        while a != s:
            a = pred[a]
            path.append(a)
        path.reverse()

        return (value, marqued, path)



class Delaunay(Graph):

    def __init__(self, n):
        Graph.__init__(self, n)
        
        #gererate n randoms points
        self.x = 100.0*np.random.random(n)
        self.y = 100.0*np.random.random(n)
        
        #make the first and last point at fixed position
        self.x[0] = 0.
        self.x[n-1] = 100.0
        self.y[0] = 0.0
        self.y[n-1] = 100.0

        #caculate the distance between points
        self.distance = np.empty((self.n, self.n))
        self.getDistance()
        
        g = nx.Graph()
        for i in range(n):
            g.add_node(str(i+1))

        #create the delaunay construct
        d = DelaunayConstruct()
        for v in range(n):
            d.addPoint((self.x[v], self.y[v]))

        #put it into a graph
        triangles = d.exportTriangles()
        for (s1, s2, s3) in triangles:
            #print(s1, s2, s3)
            if not(s1 in self.edges[s2]):
                g.add_edge(str(s1+1), str(s2+1),weight =self.distance[s1, s2],proba=random.uniform(0,1))
                self.addEdge(s1, s2, self.distance[s1, s2])
            if not(s2 in self.edges[s3]):
                g.add_edge(str(s2+1), str(s3+1),weight =self.distance[s2, s3],proba=random.uniform(0,1))
                self.addEdge(s2, s3, self.distance[s2, s3])
            if not(s3 in self.edges[s1]):
                g.add_edge(str(s3+1), str(s1+1),weight =self.distance[s3, s1],proba=random.uniform(0,1))
                self.addEdge(s3, s1, self.distance[s3, s1])
        
        self.graph=g
        self.setSource(0)
        self.setSink(n-1)
        self.title = "Delaunay graphs"

    def getDistance(self):
        for i in range(self.n):
            xi = self.x[i]
            yi = self.y[i]
            self.distance[i] = np.sqrt(np.square(self.x - xi) + np.square(self.y - yi))

    def getBlockedEdges(self, n, r=5.0):
        cx = (100.0-2*r)*np.random.random(n)+r
        cy = (100.0-2*r)*np.random.random(n)+r
        blockedPoint = []
        for i in range(self.n):
            for j in range(n):
                if np.sqrt((cx[j]-self.x[i])**2 + (cy[j]-self.y[i])**2) < r:
                    blockedPoint.append(i)
                    break
        blocked = []
        for i in blockedPoint:
            for j in self.edges[i]:
                if not((j, i) in blocked):
                    blocked.append((i, j))
        return blocked, cx, cy

"""Adapted code from Jose M. Espadero ( http://github.com/jmespadero/pyDelaunay2D )"""

class DelaunayConstruct:

    def __init__(self, center=(0, 0), radius=9999):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def addPoint(self, p):
        """Add a point to the current DT, and refine it using Bowyer-Watson.
        """
        p = np.asarray(p)
        idx = len(self.coords)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

