import random

class Particle:
    def __init__(self, x0, num_dimensions):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.num_dimensions = num_dimensions

        for i in range(0, self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(random.uniform(-1,1))#x0[i])

    def evaluate(self, costFunc, shape, activations, X, Y):
        self.err_i=costFunc(self.position_i, shape, activations, X, Y)

        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    def update_velocity(self, pos_best_g):
        w=0.5      # constant inertia weight
        c1=1        # cognative constant
        c2=1.5        # social constant

        for i in range(0, self.num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive+vel_social

    def update_position(self,bounds):
        for i in range(0, self.num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # # adjust maximum position if necessary
            # if self.position_i[i]>bounds[i][1]:
            #     self.position_i[i]=bounds[i][1]

            # # adjust minimum position if neseccary
            # if self.position_i[i] < bounds[i][0]:
            #     self.position_i[i]=bounds[i][0]
                