import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Trajectory(nn.Module):
    def __init__(self, coeffs, dts):
        super(Trajectory, self).__init__()
        self.coeffs = coeffs # [n_pieces, n_coeffs, n_dims]
        self.dts = dts # [n_pieces]
        # cumulate dts
        self.ts = dts.cumsum(0).detach().numpy()
        self.ts = np.concatenate([[0], self.ts])

    def pos(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        pos = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            pos.append(self._pos(t_ - self.ts[i], self.coeffs[i]))
        pos = torch.cat(pos, 0)
        return pos
    
    def _pos(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute pos for each dimension
        T = torch.cat([t[:, None]**i for i in range(len(coeffs))], 1)
        pos = T @ coeffs
        return pos
    
    def vel(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        vel = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            vel.append(self._vel(t_ - self.ts[i], self.coeffs[i]))
        vel = torch.cat(vel, 0)
        return vel
    
    def _vel(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute vel for each dimension
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-1)], 1)
        vel = T @ (torch.arange(1, len(coeffs)).float()[:, None] * coeffs[1:])
        return vel
    
    def acc(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        acc = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            acc.append(self._acc(t_ - self.ts[i], self.coeffs[i]))
        acc = torch.cat(acc, 0)
        return acc
    
    def _acc(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute acc for each dimension
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-2)], 1)
        acc = T @ (torch.arange(2, len(coeffs)).float()[:, None] * (torch.arange(1, len(coeffs)-1).float()[:, None] * coeffs[2:]))
        return acc
    
    def jerk(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        jerk = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            jerk.append(self._jerk(t_ - self.ts[i], self.coeffs[i]))
        jerk = torch.cat(jerk, 0)
        return jerk
    
    def _jerk(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute jerk for each dimension
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-3)], 1)
        jerk = T @ (torch.arange(3, len(coeffs)).float()[:, None] * (torch.arange(2, len(coeffs)-1).float()[:, None] * (torch.arange(1, len(coeffs)-2).float()[:, None] * coeffs[3:])))
        return jerk
    
    def snap(self, t):
        """
        Compute snap (4th derivative) at given time samples.
        
        Parameters
        ----------
        t : torch.Tensor
            Time samples [n_samples]
        
        Returns
        -------
        snap : torch.Tensor
            Snap values [n_samples, n_dims]
        """
        # split t into intervals based on t >= ts_i and t < ts_i+1
        snap = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            if len(t_) > 0:
                snap.append(self._snap(t_ - self.ts[i], self.coeffs[i]))
        if len(snap) > 0:
            snap = torch.cat(snap, 0)
        else:
            snap = torch.zeros(0, self.coeffs.shape[-1])
        return snap
    
    def _snap(self, t, coeffs):
        """
        Compute snap for a single segment.
        
        For polynomial p(t) = c0 + c1*t + ... + cn*t^n:
        snap(t) = p''''(t) = 4!*c4 + 5!/1!*c5*t + 6!/2!*c6*t^2 + ...
        
        Parameters
        ----------
        t : torch.Tensor
            Time samples within segment [n_samples]
        coeffs : torch.Tensor
            Polynomial coefficients [n_coeffs, n_dims]
        
        Returns
        -------
        snap : torch.Tensor
            Snap values [n_samples, n_dims]
        """
        if len(coeffs) < 5:
            # Snap is zero for polynomials of order < 4
            return torch.zeros(len(t), coeffs.shape[-1])
        
        # Compute snap: 4th derivative
        # d^4/dt^4 [c_k * t^k] = k*(k-1)*(k-2)*(k-3) * c_k * t^(k-4)
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-4)], 1)
        
        # Factorial coefficients: k * (k-1) * (k-2) * (k-3) for k >= 4
        k = torch.arange(4, len(coeffs)).float()
        factorial_4 = k * (k-1) * (k-2) * (k-3)
        
        snap = T @ (factorial_4[:, None] * coeffs[4:])
        return snap
    
    def plot(self):
        t_ = torch.linspace(0, self.ts[-1], 1000)
        pos = self.pos(t_).detach().numpy()
        vel_abs = self.vel(t_).norm(dim=-1).detach().numpy()
        plt.figure()
        cmap = plt.get_cmap('viridis')
        colors = cmap(vel_abs/vel_abs.max())
        plt.subplot(121)
        plt.title("Position")
        plt.scatter(pos[:, 0], pos[:, 1], c=colors)
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.subplot(122)
        plt.title("Velocity")
        plt.plot(t_[:-1], vel_abs)
        for ts in self.ts:
            plt.axvline(ts, c='r')
        plt.xlabel("Time")