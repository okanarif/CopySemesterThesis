import torch
import torch.nn as nn
from .trajectory import Trajectory

# initial constraints: position, velocity, acceleration
# polynomial order: 5 (jerk)

class MINCO_S3NU(nn.Module):
    def __init__(self, N):
        super(MINCO_S3NU, self).__init__()
        self.N = N

    def set_constant_time_allocation(self, t):
        # create a matrix T [B, N, 6] with t^0, t^1, t^2, t^3, t^4, t^5
        T = torch.cat([torch.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1)
        self.T = T

        N = self.N
        B = t.shape[0]
        
        A = torch.zeros(B, 6 * N, 6 * N)
        iC = torch.zeros(6 * N, dtype=torch.bool) # mask for open constraints

        # inital constraints
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 2.0
        iC[0:3] = True

        for i in range(N-1):
            # jerk continuity
            A[:, 6*i+3, 6*i+3] = 6.0
            A[:, 6*i+3, 6*i+4] = 24.0 * T[:, i, 1]
            A[:, 6*i+3, 6*i+5] = 60.0 * T[:, i, 2]
            A[:, 6*i+3, 6*i+9] = -6.0

            # snap continuity
            A[:, 6*i+4, 6*i+4] = 24.0
            A[:, 6*i+4, 6*i+5] = 120.0 * T[:, i, 1]
            A[:, 6*i+4, 6*i+10] = -24.0
            
            # position constaint
            A[:, 6*i+5, 6*i] = T[:, i, 0]
            A[:, 6*i+5, 6*i+1] = T[:, i, 1]
            A[:, 6*i+5, 6*i+2] = T[:, i, 2]
            A[:, 6*i+5, 6*i+3] = T[:, i, 3]
            A[:, 6*i+5, 6*i+4] = T[:, i, 4]
            A[:, 6*i+5, 6*i+5] = T[:, i, 5]

            iC[6*i+5] = True

            # position continuity
            A[:, 6*i+6, 6*i] = T[:, i, 0]
            A[:, 6*i+6, 6*i+1] = T[:, i, 1]
            A[:, 6*i+6, 6*i+2] = T[:, i, 2]
            A[:, 6*i+6, 6*i+3] = T[:, i, 3]
            A[:, 6*i+6, 6*i+4] = T[:, i, 4]
            A[:, 6*i+6, 6*i+5] = T[:, i, 5]
            A[:, 6*i+6, 6*i+6] = -T[:, i, 0]

            # velocity continuity
            A[:, 6*i+7, 6*i+1] = T[:, i, 0]
            A[:, 6*i+7, 6*i+2] = 2.0 * T[:, i, 1]
            A[:, 6*i+7, 6*i+3] = 3.0 * T[:, i, 2]
            A[:, 6*i+7, 6*i+4] = 4.0 * T[:, i, 3]
            A[:, 6*i+7, 6*i+5] = 5.0 * T[:, i, 4]
            A[:, 6*i+7, 6*i+7] = -1.0

            # acceleration continuity
            A[:, 6*i+8, 6*i+2] = 2.0
            A[:, 6*i+8, 6*i+3] = 6.0 * T[:, i, 1]
            A[:, 6*i+8, 6*i+4] = 12.0 * T[:, i, 2]
            A[:, 6*i+8, 6*i+5] = 20.0 * T[:, i, 3]
            A[:, 6*i+8, 6*i+8] = -2.0

        # final constraints
        # position
        A[:, 6*N-3, 6*N-6] = T[:, N-1, 0]
        A[:, 6*N-3, 6*N-5] = T[:, N-1, 1]
        A[:, 6*N-3, 6*N-4] = T[:, N-1, 2]
        A[:, 6*N-3, 6*N-3] = T[:, N-1, 3]
        A[:, 6*N-3, 6*N-2] = T[:, N-1, 4]
        A[:, 6*N-3, 6*N-1] = T[:, N-1, 5]

        # velocity
        A[:, 6*N-2, 6*N-5] = T[:, N-1, 0]
        A[:, 6*N-2, 6*N-4] = 2.0 * T[:, N-1, 1]
        A[:, 6*N-2, 6*N-3] = 3.0 * T[:, N-1, 2]
        A[:, 6*N-2, 6*N-2] = 4.0 * T[:, N-1, 3]
        A[:, 6*N-2, 6*N-1] = 5.0 * T[:, N-1, 4]

        # acceleration
        A[:, 6*N-1, 6*N-4] = 2.0
        A[:, 6*N-1, 6*N-3] = 6.0 * T[:, N-1, 1]
        A[:, 6*N-1, 6*N-2] = 12.0 * T[:, N-1, 2]
        A[:, 6*N-1, 6*N-1] = 20.0 * T[:, N-1, 3]

        iC[6*N-3:6*N] = True

        # solve the linear system
        A_inv = torch.linalg.inv(A)  # placeholder for later use

        # linear mapping from waypoint constraints (initial+final+waypoint) to polynomial coefficients (constant linear mapping for fixed time allocation)
        self.A_map = A_inv[:, :, iC]

    def solve_linear_fixedtime(self, headPVA, tailPVA, points):
        # headPVA: [B, 3, 3]
        # tailPVA: [B, 3, 3]
        # points: [B, N-1, 3]
        # return: [B, 6N, 3] polynomial coefficients

        assert headPVA.shape[0] == tailPVA.shape[0] == points.shape[0], f"Tensors must have the same batch size. Got {headPVA.shape[0]}, {tailPVA.shape[0]}, {points.shape[0]}"
        
        N = self.N
        B = headPVA.shape[0]
        
        # stack headPVA, tailPVA, points into b [B, n_constraints, 3]
        b = torch.concat([
            headPVA,
            points,
            tailPVA
        ], dim=1).float()  # [B, 6N, 3]

        x = torch.bmm(self.A_map, b)  # [B, 6N, 3]
        self.x = x

    def solve_linear(self, headPVA, tailPVA, points, t):
        # headPVA: [B, 3, 3]
        # tailPVA: [B, 3, 3]
        # points: [B, N-1, 3]
        # t: [B, N, 1]
        # return: [B, 1]

        # create a matrix T [B, N, 6] with t^0, t^1, t^2, t^3, t^4, t^5
        T = torch.cat([torch.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1)
        self.T = T

        assert headPVA.shape[0] == tailPVA.shape[0] == points.shape[0] == t.shape[0], f"Tensors must have the same batch size. Got {headPVA.shape[0]}, {tailPVA.shape[0]}, {points.shape[0]}, {t.shape[0]}"
        
        N = self.N
        B = headPVA.shape[0]
        
        A = torch.zeros(B, 6 * N, 6 * N)
        b = torch.zeros(B, 6 * N, 3)
        iC = torch.zeros(6 * N, dtype=torch.bool) # mask for open constraints

        # inital constraints
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 2.0
        b[:, 0] = headPVA[:, 0]
        b[:, 1] = headPVA[:, 1]
        b[:, 2] = headPVA[:, 2]
        iC[0:3] = True

        for i in range(N-1):
            # jerk continuity
            A[:, 6*i+3, 6*i+3] = 6.0
            A[:, 6*i+3, 6*i+4] = 24.0 * T[:, i, 1]
            A[:, 6*i+3, 6*i+5] = 60.0 * T[:, i, 2]
            A[:, 6*i+3, 6*i+9] = -6.0

            # snap continuity
            A[:, 6*i+4, 6*i+4] = 24.0
            A[:, 6*i+4, 6*i+5] = 120.0 * T[:, i, 1]
            A[:, 6*i+4, 6*i+10] = -24.0
            
            # position constaint
            A[:, 6*i+5, 6*i] = T[:, i, 0]
            A[:, 6*i+5, 6*i+1] = T[:, i, 1]
            A[:, 6*i+5, 6*i+2] = T[:, i, 2]
            A[:, 6*i+5, 6*i+3] = T[:, i, 3]
            A[:, 6*i+5, 6*i+4] = T[:, i, 4]
            A[:, 6*i+5, 6*i+5] = T[:, i, 5]

            b[:, 6*i+5] = points[:, i]
            iC[6*i+5] = True

            # position continuity
            A[:, 6*i+6, 6*i] = T[:, i, 0]
            A[:, 6*i+6, 6*i+1] = T[:, i, 1]
            A[:, 6*i+6, 6*i+2] = T[:, i, 2]
            A[:, 6*i+6, 6*i+3] = T[:, i, 3]
            A[:, 6*i+6, 6*i+4] = T[:, i, 4]
            A[:, 6*i+6, 6*i+5] = T[:, i, 5]
            A[:, 6*i+6, 6*i+6] = -T[:, i, 0]

            # velocity continuity
            A[:, 6*i+7, 6*i+1] = T[:, i, 0]
            A[:, 6*i+7, 6*i+2] = 2.0 * T[:, i, 1]
            A[:, 6*i+7, 6*i+3] = 3.0 * T[:, i, 2]
            A[:, 6*i+7, 6*i+4] = 4.0 * T[:, i, 3]
            A[:, 6*i+7, 6*i+5] = 5.0 * T[:, i, 4]
            A[:, 6*i+7, 6*i+7] = -1.0

            # acceleration continuity
            A[:, 6*i+8, 6*i+2] = 2.0
            A[:, 6*i+8, 6*i+3] = 6.0 * T[:, i, 1]
            A[:, 6*i+8, 6*i+4] = 12.0 * T[:, i, 2]
            A[:, 6*i+8, 6*i+5] = 20.0 * T[:, i, 3]
            A[:, 6*i+8, 6*i+8] = -2.0

        # final constraints
        # position
        A[:, 6*N-3, 6*N-6] = T[:, N-1, 0]
        A[:, 6*N-3, 6*N-5] = T[:, N-1, 1]
        A[:, 6*N-3, 6*N-4] = T[:, N-1, 2]
        A[:, 6*N-3, 6*N-3] = T[:, N-1, 3]
        A[:, 6*N-3, 6*N-2] = T[:, N-1, 4]
        A[:, 6*N-3, 6*N-1] = T[:, N-1, 5]

        # velocity
        A[:, 6*N-2, 6*N-5] = T[:, N-1, 0]
        A[:, 6*N-2, 6*N-4] = 2.0 * T[:, N-1, 1]
        A[:, 6*N-2, 6*N-3] = 3.0 * T[:, N-1, 2]
        A[:, 6*N-2, 6*N-2] = 4.0 * T[:, N-1, 3]
        A[:, 6*N-2, 6*N-1] = 5.0 * T[:, N-1, 4]

        # acceleration
        A[:, 6*N-1, 6*N-4] = 2.0
        A[:, 6*N-1, 6*N-3] = 6.0 * T[:, N-1, 1]
        A[:, 6*N-1, 6*N-2] = 12.0 * T[:, N-1, 2]
        A[:, 6*N-1, 6*N-1] = 20.0 * T[:, N-1, 3]

        b[:, 6*N-3] = tailPVA[:, 0]
        b[:, 6*N-2] = tailPVA[:, 1]
        b[:, 6*N-1] = tailPVA[:, 2]

        iC[6*N-3:6*N] = True

        # solve the linear system
        x = torch.linalg.solve(A, b)

        self.A = A
        self.x = x

    def get_energy(self):
        energy = 0.0
        for i in range(self.N-1):
            energy += 36.0 * self.x[:, 6*i+3].pow(2).sum() * self.T[:, i, 1] \
                    + 144.0 * (self.x[:, 6*i+4] * self.x[:, 6*i+3]).sum() * self.T[:, i, 2] \
                    + 192.0 * self.x[:, 6*i+4].pow(2).sum() * self.T[:, i, 3] \
                    + 240.0 * (self.x[:, 6*i+5] * self.x[:, 6*i+3]).sum() * self.T[:, i, 3] \
                    + 720.0 * (self.x[:, 6*i+5] * self.x[:, 6*i+4]).sum() * self.T[:, i, 4] \
                    + 720.0 * self.x[:, 6*i+5].pow(2).sum() * self.T[:, i, 5]
        return energy

    # forward pass
    def forward(self, headPVA, tailPVA, points, t):
        self.solve_linear(headPVA, tailPVA, points, t)
        energy = self.get_energy()
        return energy
    
    def get_trajectory(self):
        trajs = []
        for i in range(self.x.shape[0]):
            coeffs = self.x[i].view(self.N, 6, 3)
            traj = Trajectory(coeffs, self.T[i, :, 1])
            trajs.append(traj)
        return trajs