import random

class Particle:
    def __init__(self, weights, num_dimensions, config):
        self.position_i = []
        self.velocity_i = []
        self.pos_best_i = []
        self.err_best_i = -1
        self.err_i = -1
        self.num_dimensions = num_dimensions
        self.ine_cst = config["inertia_cst"]
        self.co_cst = config["cognative_cst"]
        self.so_cst = config["social_cst"]

        for i in range(0, self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(weights[i])

    def evaluate(self, costFunc, shape, activations, X, Y):
        self.err_i = costFunc(self.position_i, shape, activations, X, Y)

        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    def update_velocity(self, pos_best_g):
        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = self.co_cst * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = self.so_cst * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = self.ine_cst * self.velocity_i[i] + vel_cognitive + vel_social

    def update_position(self, bounds):
        for i in range(0,self.num_dimensions):
            self.position_i[i] = self.position_i[i]+self.velocity_i[i]

            if self.position_i[i] > bounds[1]:
                self.position_i[i] = bounds[1]

            if self.position_i[i] < bounds[0]:
                self.position_i[i] = bounds[0]