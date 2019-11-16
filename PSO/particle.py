import random

class Particle:
    def __init__(self, x0, num_dimensions, config):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.num_dimensions = num_dimensions
        self.w = config["inertia_cst"]
        self.c1 = config["cognative_cst"]
        self.c2 = config["social_cst"]

        for i in range(0, self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(random.uniform(-1,1))  #x0[i]) # TODO init pos

    def evaluate(self, costFunc, shape, activations, X, Y):
        self.err_i=costFunc(self.position_i, shape, activations, X, Y)

        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    def update_velocity(self, pos_best_g):
        for i in range(0, self.num_dimensions):
            r1=random.random()
            r2=random.random() # TODO check why random

            vel_cognitive = self.c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = self.c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = self.w * self.velocity_i[i] + vel_cognitive+vel_social

    def update_position(self):
        for i in range(0, self.num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]