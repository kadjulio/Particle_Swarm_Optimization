import functions.activitions as act
configs = [
    { "shape": [4, 4, 1], "activations": [act.Sigmoid, act.Sigmoid, act.Tanh], "nb_particles": 100, "max_iter": 10, "inertia_cst": 0.5, "cognative_cst": 1, "social_cst": 1.5, "bounds": [-2.0, 2.0]},
    { "shape": [4, 4, 1], "activations": [act.Relu, act.Relu, act.Sigmoid], "nb_particles": 100, "max_iter": 10, "inertia_cst": 0.5, "cognative_cst": 1, "social_cst": 1.5, "bounds": [-2.0, 2.0]},
    { "shape": [4, 4, 1], "activations": [act.Sigmoid, act.Sigmoid, act.Tanh], "nb_particles": 30, "max_iter": 100, "inertia_cst": 0.7298, "cognative_cst": 1.4960, "social_cst": 1.4960, "bounds": [-2.0, 2.0]},
]