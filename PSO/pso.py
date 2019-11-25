from .particle import Particle
from math import sqrt


class PSO():
###
# https://towardsdatascience.com/particle-swarm-optimisation-in-machine-learning-b01b1d2ad8a8
###
    def __init__(self, costFunc, weights, shape, activations, X, Y, config):

        self.num_dimensions=len(weights)
        self.err_best_g=-1
        self.pos_best_g=[]
        self.swarm=[]
    
        for i in range(0, config["nb_particles"]):
            self.swarm.append(Particle(weights, self.num_dimensions, config))

        for i in range(config["max_iter"]):
            # if (i % int(sqrt(config["max_iter"])) == 0):
            #     print (i, self.err_best_g)

            for j in range(0, config["nb_particles"]):
                self.swarm[j].evaluate(costFunc, shape, activations, X, Y)

                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(self.swarm[j].position_i)
                    self.err_best_g=float(self.swarm[j].err_i)

            for j in range(0, config["nb_particles"]):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(config["bounds"])

    def get_pos_best_g(self):
        return self.pos_best_g