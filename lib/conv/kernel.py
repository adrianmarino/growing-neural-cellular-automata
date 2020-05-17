import torch as t

SOLVER_X = t.tensor([
    [-1., 0., +1.],
    [-2., 0., +2.],
    [-1., 0., +1.],
])

SOLVER_Y = SOLVER_X.t()

IDENTITY = t.tensor([
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.],
])
