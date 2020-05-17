import torch as t

SOLVER_X = t.tensor([
    [-1, 0, +1],
    [-2, 0, +2],
    [-1, 0, +1],
]).type(t.float)
SOLVER_Y = SOLVER_X.t()

IDENTITY = t.eye(3).type(t.float)
