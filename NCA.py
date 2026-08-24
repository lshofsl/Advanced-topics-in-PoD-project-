from multiprocessing import dummy
from sys import prefix

import torch 
import torch.nn as nn
import torch.nn.functional as F

def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")


def perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    if mask_n != 0:
        n = x.shape[1]
        padd = torch.zeros((x.shape[0], 3 * mask_n, x.shape[2], x.shape[3]), device="cuda:0")
        obs = perchannel_conv(x[:, 0:n - mask_n], filters)
        return torch.cat((x, obs, padd), dim=1)
    else:
        obs = perchannel_conv(x, filters)
        return torch.cat((x,obs), dim = 1 )

def masked_perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    mask = torch.zeros_like(x)
    mask[:,0:x.shape[1]- mask_n,...] = 1
    x_masked = x*mask


    obs = perchannel_conv(x_masked,filters)
    return torch.cat((x,obs), dim = 1 )


def reduced_perception(x, mask_n=0):
    filters = torch.stack([sobel_x, sobel_x.T, lap])
    x_redu = x[:,0:x.shape[1]-mask_n]
    obs = perchannel_conv(x_redu,filters)
    return torch.cat((x,obs), dim = 1 )
    

class EnergyNCA(nn.Module):

    def __init__(self, chn=16):
        super().__init__()
        self.chn = chn
        self.perceive_dim = chn * 4  # 64 channels: [Identity, SobelX, SobelY, Laplacian]

        # 1. Local Interaction Matrix W (64 x 64)
        self.W = nn.Parameter(
            torch.randn(self.perceive_dim, self.perceive_dim) * 0.01
        )
        self.beta = nn.Parameter(torch.tensor(0.01))

        # 2. Linear Field / Bias h (64,)
        self.h = nn.Parameter(torch.zeros(self.perceive_dim))
        with torch.no_grad():
            self.h[3] = 0.5  # Strong alpha initialization

        # Perception filters
        sobel_x = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            / 8.0
        )
        laplacian = (
            torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])
            / 8.0
        )

        self.register_buffer(
            "Kx", sobel_x.view(1, 1, 3, 3).repeat(chn, 1, 1, 1)
        )
        self.register_buffer(
            "Ky", sobel_x.T.view(1, 1, 3, 3).repeat(chn, 1, 1, 1)
        )
        self.register_buffer(
            "Klap", laplacian.view(1, 1, 3, 3).repeat(chn, 1, 1, 1)
        )

        self.log_eta = nn.Parameter(torch.tensor(-3.8))

    def cohen_grossberg_damping(self, s, beta):
        return 0.5 * torch.tanh(beta * s) ** 2

    def perceive(self, s):
        s_padded = F.pad(s, [1, 1, 1, 1], mode="constant")
        sx = F.conv2d(s_padded, self.Kx, groups=self.chn)
        sy = F.conv2d(s_padded, self.Ky, groups=self.chn)
        slap = F.conv2d(s_padded, self.Klap, groups=self.chn)

        return torch.cat([s, sx, sy, slap], dim=1)

    def _get_constrained_W(self):
        A = self.W
        W_sym = -0.5 * (A + A.T)
        return W_sym

    def energy(self, x):
        s = x[:, : self.chn, ...]
        p = self.perceive(s)
        beta = F.softplus(self.beta)
        W_sym = self._get_constrained_W()

        p_W = torch.einsum("ij,bjhw->bihw", W_sym, p)
        e_quad = -0.5 * (p * p_W).sum(dim=1)
        h_clamped = torch.clamp(self.h, -0.05, 0.05)
        e_lin = -(p * h_clamped.view(1, -1, 1, 1)).sum(dim=1)

        diffusion = self.cohen_grossberg_damping(s, beta).sum(dim=1) 

        return (e_quad + e_lin + diffusion).sum(dim=[1, 2])

    def energy_gradient(self, x):
        s = x[:, : self.chn, ...]
        p = self.perceive(s)
        beta = F.softplus(self.beta)

        W_sym = self._get_constrained_W()
        h_clamped = torch.clamp(self.h, -0.05, 0.05)
        p_transformed = torch.einsum(
            "ij,bjhw->bihw", W_sym, p
        ) + h_clamped.view(1, -1, 1, 1)

        # Slice channel blocks (each is chn long)
        d_id = p_transformed[:, : self.chn]
        d_sx = p_transformed[:, self.chn : 2 * self.chn]
        d_sy = p_transformed[:, 2 * self.chn : 3 * self.chn]
        d_slap = p_transformed[:, 3 * self.chn :]

        s_padded_dsx = F.pad(d_sx, [1, 1, 1, 1], mode="constant")
        s_padded_dsy = F.pad(d_sy, [1, 1, 1, 1], mode="constant")
        s_padded_dslap = F.pad(d_slap, [1, 1, 1, 1], mode="constant")

        # Transposed convolution terms for coupling
        # Note: Kx and Ky swap sign due to anti-symmetric spatial derivative (K^T = -K)
        # Klap preserves sign because the Laplacian kernel is symmetric (K^T = K)
        grad_coupling_bias = (
            d_id
            - F.conv2d(s_padded_dsx, self.Kx, groups=self.chn)
            - F.conv2d(s_padded_dsy, self.Ky, groups=self.chn)
            + F.conv2d(s_padded_dslap, self.Klap, groups=self.chn)
        )

        fs = torch.tanh(beta * s)
        grad_damping = beta * (1.0 - fs**2) * fs

        total = grad_coupling_bias + grad_damping
        return torch.clamp(total, -1.0, 1.0)

    def get_alive_mask(self, x):
        alpha = x[:, 3:4, :, :]
        padded_alpha = F.pad(alpha, pad=[1, 1, 1, 1], mode="constant")
        return F.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x).float()
        batch_n, _, h, w = x.shape
        eta = F.softplus(self.log_eta)

        # Direct Negative Energy Gradient Descent
        grad_E = self.energy_gradient(x)
        correction = eta * grad_E

        update_mask = (
            torch.rand(batch_n, 1, h, w, device=x.device) < update_rate
        )

        # Out-of-place state update
        x_raw = x + correction * update_mask * pre_life_mask

        # Out-of-place clamping to prevent exploding gradients
        rgba = torch.clamp(x_raw[:, :4, ...], 0.0, 1.0)
        hidden = torch.tanh(x_raw[:, 4:, ...])
        x_clamped = torch.cat([rgba, hidden], dim=1)

        post_life_mask = self.get_alive_mask(x_clamped).float()
        return x_clamped * post_life_mask



class HYBRID_NCA(torch.nn.Module):
    def __init__(self, chn=16, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3 * (chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()


        self.v_dim = 4
        self.h_dim = chn - self.v_dim
        self.W = nn.Parameter(torch.randn(chn, chn) * 0.01)    # Now the matrix is state-state size and do not depend on the perception vector 
        self.eta = torch.nn.Parameter(torch.tensor(0.1))  
        self.beta = nn.Parameter(torch.tensor(0.01))
        self.h = nn.Parameter(torch.zeros(chn))
        

    def cohen_grossberg_damping(self, s, beta):
        z = beta * s
        abs_z = torch.abs(z)
        log_cosh = abs_z + torch.log1p(torch.exp(-2.0 * abs_z)) - 0.69314718
        return 0.5 * (s ** 2) - (1.0 / beta) * log_cosh
    
    def get_alive_mask(self,x):
        alpha = x[:, 3:4, :, :] 
        padded_alpha = torch.nn.functional.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def _get_constrained_W(self):
        A = self.W
        W_sym = 0.5 * (A + A.T)
        return W_sym
        
    def MLP(self, x): 
        y = reduced_perception(x) # compute spatial perception
        y = self.w2(F.relu(self.w1(y))) # pass tensor y through layers
        return y
        
    def energy(self, x):
        s = x[:, :self.chn, ...]
        beta = F.softplus(self.beta)  
        # The couping matrix remains symmetric always
        W_sym = self._get_constrained_W()
        # We constraint the bias factor to not overgrowht the energy 
        h_clamped = torch.clamp(self.h, -0.1, 0.1)
        y = self.MLP(s)

        s_W = torch.einsum('ij,bjhw->bihw', W_sym, s)
        #Negative term
        e_quad = -0.5 * (s * s_W).sum(dim=1)
        #Negative term
        e_lin = -(s * h_clamped.view(1, -1, 1, 1)).sum(dim=1)
        #Negative term
        e_per = -(y * s).sum(dim=1)
        #Positive term
        s_rgba = s[:, :4]
        s_hidden = s[:, 4:]
        beta_rgba = 0.7
        beta_hidden = 1.0 
        damping_rgba = self.cohen_grossberg_damping(s_rgba, beta_rgba)   # [B, 4, H, W]
        damping_hidden = self.cohen_grossberg_damping(s_hidden, beta_hidden) # [B, 12, H, W]
        e_damping_channels = torch.cat([damping_rgba, damping_hidden], dim=1) # [B, 16, H, W]
        e_damping = e_damping_channels.sum(dim=1)
        #All terms are calcualted with their respective signs 
        return (e_quad + e_lin + e_damping + e_per).sum(dim=[1, 2])

    def energy_gradient(self, x):
        """Calculates exact dE/ds (Energy Gradient)."""
        s = x[:, :self.chn, ...]
        beta = F.softplus(self.beta)
        W_sym = self._get_constrained_W()
        h_clamped = torch.clamp(self.h, -0.1, 0.1)

        y = self.MLP(x)

        # 1. Coupling and Bias Gradient: -W_sym @ s - h
        s_W = torch.einsum('ij,bjhw->bihw', W_sym, s)
        grad_coupling_bias = -s_W - h_clamped.view(1, -1, 1, 1)

        # 2. Perception Field Gradient: -y
        grad_perc = -y


        s_rgba = s[:, :4]
        s_hidden = s[:, 4:]
        beta_rgba = 0.7  
        beta_hidden = 1.0 

        damp_rgba = s_rgba - torch.tanh(beta_rgba * s_rgba)
        damp_hidden = s_hidden - torch.tanh(beta_hidden * s_hidden)

        grad_damping = torch.cat([damp_rgba, damp_hidden], dim=1)
        # Exact Analytical Gradient dE/ds
        dE_ds = grad_coupling_bias + grad_perc + grad_damping
    
        return torch.clamp(dE_ds, -1.0, 1.0)


    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        y = self.MLP(x)
        b, c, h, w = y.shape
        energy_grad =  self.energy_gradient(x)
        update_mask = (torch.rand(b, 1, h, w, device=x.device) < update_rate)
        
        eta = 0.05 * torch.sigmoid(self.eta)
        #The update state now is completly guided by the energy_grad but for the 
        #perception propery we add this state 
        dx = -eta * energy_grad * update_mask * pre_life_mask #Differential state to be updated 
        x_update = x + dx 

    
        v_part = x_update[:, :self.v_dim, ...]
        h_part = torch.tanh(x_update[:, 4:, ...])
        x_update = torch.cat([v_part, h_part], dim=1)

        post_life_mask = self.get_alive_mask(x_update)
        x_final = x_update * post_life_mask
        return x_final





        
        

