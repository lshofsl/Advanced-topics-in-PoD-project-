from multiprocessing import dummy
from sys import prefix

import torch 
import torch.nn as nn
import torch.nn.functional as F
def perchannel_conv(x, filters):
    """
    Applies filters independently to each channel in x.
    x: (B, C, H, W)
    filters: (K, 1, 3, 3) where K is number of filter kernels (3)
    Returns: (B, C * K, H, W)
    """
    b, ch, h, w = x.shape
    k = filters.shape[0]  # 3 kernels
    
    # 1. Expand filters to apply across ALL channels: (K * C, 1, 3, 3)
    # This creates K filters for each of the C input channels
    weight = filters.repeat(ch, 1, 1, 1)  
    
    # 2. Circular padding for periodic boundaries
    x_padded = F.pad(x, [1, 1, 1, 1], mode='circular')
    
    # 3. Depthwise / Grouped Convolution (groups=ch)
    # Output shape: (B, C * K, H, W)
    out = F.conv2d(x_padded, weight, groups=ch)
    
    return out

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


def reduced_perception(x, filters, mask_n=0):
    """
    Applies perchannel_conv using pass-through filters.
    x: State tensor (B, C, H, W)
    filters: Filter tensor stack (3, 1, 3, 3)
    """
    x_redu = x[:, 0 : x.shape[1] - mask_n]
    obs = perchannel_conv(x_redu, filters)
    return torch.cat((x, obs), dim=1)


class EnergyNCA(nn.Module):
    def __init__(self, chn=16):
        super().__init__()
        self.chn = chn
        self.perceive_dim = chn * 4  # 64 channels (Identity + 3 spatial derivatives)

        # 1. Local Interaction Matrix W
        self.W = nn.Parameter(torch.randn(self.perceive_dim, self.perceive_dim) * 0.01)
        self.beta = nn.Parameter(torch.tensor(0.01))

        # 2. Linear Field / Bias h
        self.h = nn.Parameter(torch.zeros(self.perceive_dim))
        with torch.no_grad():
            self.h[3] = 0.5  # Strong alpha initialization

        self.log_eta = nn.Parameter(torch.tensor(-3.8))

        # 3. Spatial Filter Buffers
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32)
        lap = torch.tensor([[1.0, 2.0, 1.0],
                            [2.0, -12.0, 2.0],
                            [1.0, 2.0, 1.0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def get_filters(self):
        sobel_y = self.sobel_x.transpose(-1, -2)
        return torch.cat([self.sobel_x, sobel_y, self.lap], dim=0)

    def cohen_grossberg(self, s, beta):
        fs = torch.tanh(beta * s)
        return s * fs - s + (1.0 / beta) * torch.log(1.0 + fs)

    def _get_constrained_W(self):
        A = self.W
        return 0.5 * (A + A.T)

    def energy(self, x):
        s = x[:, :self.chn, ...]
        filters = self.get_filters()
        p = reduced_perception(s, filters)
        beta = F.softplus(self.beta)

        W_sym = self._get_constrained_W()
        h_clamped = torch.clamp(self.h, -0.05, 0.05)

        p_W = torch.einsum('ij,bjhw->bihw', W_sym, p)
        e_quad = -0.5 * (p * p_W).sum(dim=1)
        e_lin = -(p * h_clamped.view(1, -1, 1, 1)).sum(dim=1)
        diffusion = self.cohen_grossberg(s, beta).sum(dim=1)

        return (e_quad + e_lin + diffusion).sum(dim=[1, 2])

    def energy_gradient(self, x, create_graph=False):
        s = x[:, :self.chn, ...]

        with torch.enable_grad():
            s_ = s if s.requires_grad else s.detach().requires_grad_(True)
            filters = self.get_filters()

            # Recompute perception features on s_
            p = reduced_perception(s_, filters)

            # Transform perceived state
            W_sym = self._get_constrained_W()
            h_clamped = torch.clamp(self.h, -0.05, 0.05)
            p_transformed = torch.einsum('ij,bjhw->bihw', W_sym, p) + h_clamped.view(1, -1, 1, 1)

            # Compute VJP: J_p^T * p_transformed
            grad_coupling_bias, = torch.autograd.grad(
                outputs=p,
                inputs=s_,
                grad_outputs=p_transformed,
                create_graph=create_graph,
                retain_graph=True
            )

        beta = F.softplus(self.beta)
        fs = torch.tanh(beta * s)
        grad_diffusion = s * beta * (1 - fs**2)

        total = grad_coupling_bias - grad_diffusion
        return torch.clamp(total, -1.0, 1.0)

    def get_alive_mask(self, x):
        alpha = x[:, 3:4, :, :]
        padded_alpha = F.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return F.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x).float()
        batch_n, _, h, w = x.shape
        eta = F.softplus(self.log_eta)

        # Direct Negative Energy Gradient Descent
        grad_E = self.energy_gradient(x)
        correction = eta * grad_E

        update_mask = (torch.rand(batch_n, 1, h, w, device=x.device) < update_rate)

        x_raw = x + correction * update_mask * pre_life_mask

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
        

    def cohen_grossberg(self, s, beta):
        fs = torch.tanh(beta * s)
        return s * fs - s + (1.0 / beta) * torch.log(1.0 + fs)

    
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
        h_clamped = torch.clamp(self.h, -0.05, 0.05)
        y = self.MLP(s)

        s_W = torch.einsum('ij,bjhw->bihw', W_sym, s)
        #Negative term
        e_quad = -0.5 * (s * s_W).sum(dim=1)
        #Negative term
        e_lin = -(s * h_clamped.view(1, -1, 1, 1)).sum(dim=1)
        #Negative term
        e_per = -(y * s).sum(dim=1)
        #Positive term
        diffusion = self.cohen_grossberg(s, beta).sum(dim=1)   
        #All terms are calcualted with their respective signs 
        return (e_quad + e_lin + diffusion + e_per).sum(dim=[1, 2])

    def energy_gradient(self, x,  create_graph=True):
        """Calculates exact dE/ds (Energy Gradient)."""
        s = x[:, :self.chn, ...]
        beta = F.softplus(self.beta)
        W_sym = self._get_constrained_W()
        h_clamped = torch.clamp(self.h, -0.05, 0.05)

        # 1. Coupling and Bias Gradient: -W_sym @ s - h
        s_W = torch.einsum('ij,bjhw->bihw', W_sym, s)
        grad_coupling_bias = -s_W - h_clamped.view(1, -1, 1, 1)

        # 2. Perception Field Gradient: torch grad to obtain the derivative of the kernels filters
        with torch.enable_grad():                                 
            s_ = s if s.requires_grad else s.detach().requires_grad_(True)
            y = self.MLP(s_)
            vjp, = torch.autograd.grad(y, s_, grad_outputs=s_, create_graph=create_graph)
        grad_perc = -y - vjp
        # 3. Cohen-Grossberg Barrier Gradient: d/ds [V_CG(s)] = s - tanh(beta * s)
        fs = torch.tanh(beta * s)
        grad_diffusion = s * beta * (1 - fs**2)

        # Exact Analytical Gradient dE/ds
        dE_ds = grad_coupling_bias + grad_perc + grad_diffusion
    
        return torch.clamp(dE_ds, -1.0, 1.0)


    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        y = self.MLP(x)
        b, c, h, w = y.shape
        energy_grad =  self.energy_gradient(x,  create_graph=True)
        update_mask = (torch.rand(b, 1, h, w, device=x.device) < update_rate)
        
        eta = 0.05 * torch.sigmoid(self.eta)
        #The update state now is completly guided by the energy_grad but for the 
        #perception propery we add this state 
        dx = -eta * energy_grad * update_mask * pre_life_mask #Differential state to be updated 
        x_update = x + dx 

        # Bound hidden channels only, leave RGBA as-is
        v_part = x_update[:, :self.v_dim, ...]
        h_part = torch.tanh(x_update[:, 4:, ...])
        x_update = torch.cat([v_part, h_part], dim=1)

        post_life_mask = self.get_alive_mask(x_update)
        x_final = x_update * post_life_mask
        return x_final






        
        

