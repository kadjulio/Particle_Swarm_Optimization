from .particle import Particle
from math import sqrt
class PSO():
    def __init__(self,costFunc,x0, shape, activations, X, Y, bounds, num_particles, maxiter):

        self.num_dimensions=len(x0)
        self.err_best_g=-1
        self.pos_best_g=[]
        self.swarm=[]
    
        for i in range(0, num_particles):
            self.swarm.append(Particle(x0, self.num_dimensions))

        i=0
        while i < maxiter:
            if (i % int(sqrt(maxiter)) == 0):
                print (i, self.err_best_g)

            for j in range(0, num_particles):
                self.swarm[j].evaluate(costFunc, shape, activations, X, Y)

                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(self.swarm[j].position_i)
                    self.err_best_g=float(self.swarm[j].err_i)

            for j in range(0, num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(bounds)
            i+=1
            # print(self.err_best_g)

        # print final results
        # print ('FINAL:')
        # print (self.pos_best_g)
        # print (self.err_best_g)
        
    def get_pos_best_g(self):
        return self.pos_best_g