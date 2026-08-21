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
    


class DummyVCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0,).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x

class MaskedCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = masked_perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x


class ReducedCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3*(chn-  mask_n), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = reduced_perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0,).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x


class GeneCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn
        self.gene_size = gene_size 
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.w1 = torch.nn.Conv2d(chn + 3 * (chn), hidden_n, 1)
        self.public = chn  - gene_size  
        self.w2 = torch.nn.Conv2d(hidden_n, self.public, 1, bias=False)
        self.w2.weight.data.zero_()
        

    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...]
        s = x[:, :self.public, ...]
        y = reduced_perception(x[:, :self.chn], 0)   #Only perceive the RGBA + hidden + genes 
        
        ##NCA state update 
        delta_s = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        ## Energy function: quadratic Hopfield form
        energy_grad = -s
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        s_update = s + (delta_s * update_mask - self.beta * energy_grad) * pre_life_mask
        x = torch.cat((s_update, gene), dim=1)
        return x



class EnergyNCA(nn.Module):
    def __init__(self, chn=16):
        super().__init__()
        self.chn = chn
        self.perceive_dim = chn * 3  # 48 channels
        
        # 1. Local Interaction Matrix W (48 x 48)
        self.W = nn.Parameter(torch.randn(self.perceive_dim, self.perceive_dim) * 0.01)
        self.beta = nn.Parameter(torch.tensor(0.01))
        
        # 2. Linear Field / Bias h (48,) - Drives spontaneous boundary growth
        self.h = nn.Parameter(torch.zeros(self.perceive_dim))
        # Pre-bias Alpha identity channel so life naturally wants to expand from neighbors
        with torch.no_grad():
            self.h[3] = 0.5 #strong alpha initialization 
        
        # Perception filters
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("Kx", sobel_x.view(1, 1, 3, 3).repeat(chn, 1, 1, 1))
        self.register_buffer("Ky", sobel_x.T.view(1, 1, 3, 3).repeat(chn, 1, 1, 1))
        
        self.log_eta = nn.Parameter(torch.tensor(-3.8))

    def cohen_grossberg(self, s, beta):
        fs = torch.tanh(beta * s)
        return s * fs - s + (1.0 / beta) * torch.log(1.0 + fs)

    def perceive(self, s):
        s_padded = F.pad(s, [1, 1, 1, 1], mode="constant")
        sx = F.conv2d(s_padded, self.Kx, groups=self.chn)
        sy = F.conv2d(s_padded, self.Ky, groups=self.chn)
    
        return torch.cat([s, sx, sy], dim=1)

    def _get_constrained_W(self):
        A = self.W
        W_sym = -0.5 * (A + A.T)
        # No self interactions 
        diag_mask = 1.0 - torch.eye(self.perceive_dim, device=A.device)
        W_no_self = W_sym * diag_mask
        return W_no_self 
        
    def energy(self, x):
        s = x[:, :self.chn, ...]
        p = self.perceive(s)
        beta = F.softplus(self.beta)   
        W_sym = self._get_constrained_W()

        p_W = torch.einsum('ij,bjhw->bihw', W_sym, p)
        e_quad = -0.5 * (p * p_W).sum(dim=1)
        h_clamped = torch.clamp(self.h, -0.05, 0.05)
        e_lin = -(p * h_clamped.view(1, -1, 1, 1)).sum(dim=1)

        diffusion = self.cohen_grossberg(s, beta).sum(dim=1)   

        return (e_quad + e_lin + diffusion).sum(dim=[1, 2])

    def energy_gradient(self, x):
        s = x[:, :self.chn, ...]
        p = self.perceive(s)
        beta = F.softplus(self.beta)

        W_sym = self._get_constrained_W()
        h_clamped = torch.clamp(self.h, -0.05, 0.05)
        p_transformed = torch.einsum('ij,bjhw->bihw', W_sym, p) + h_clamped.view(1, -1, 1, 1)

        d_id = p_transformed[:, :self.chn]
        d_sx = p_transformed[:, self.chn:2*self.chn]
        d_sy = p_transformed[:, 2*self.chn:]

        s_padded_dsx = F.pad(d_sx, [1, 1, 1, 1], mode="constant")
        s_padded_dsy = F.pad(d_sy, [1, 1, 1, 1], mode="constant")

        grad_coupling_bias = d_id \
            - F.conv2d(s_padded_dsx, self.Kx, groups=self.chn) \
            - F.conv2d(s_padded_dsy, self.Ky, groups=self.chn)

        fs = torch.tanh(beta * s)
        d_ediff_ds = beta * s * (1.0 - fs**2)
        grad_diffusion = -d_ediff_ds

        total = grad_coupling_bias + grad_diffusion
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

        update_mask = (torch.rand(batch_n, 1, h, w, device=x.device) < update_rate)
        
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
        self.W = nn.Parameter(torch.randn(chn, chn) * 0.1)    # Now the matrix is state-state size and do not depend on the perception vector 
        self.eta = torch.nn.Parameter(torch.tensor(0.2))  
        self.h = nn.Parameter(torch.zeros(chn))
        with torch.no_grad():
            self.h[3] = 0.5 #strong alpha initialization 

    
    def get_alive_mask(self,x):
        alpha = x[:, 3:4, :, :] 
        padded_alpha = torch.nn.functional.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def _get_constrained_W(self):
        A = self.W
        return -0.5 * (A @ A.T + A.T @ A) 
        
    def energy(self, x):
        s = x[:, :self.chn, ...]

        W_sym = self._get_constrained_W()

        s_W = torch.einsum('ij,bjhw->bihw', W_sym, s)
        e_quad = -0.5 * (s * s_W).sum(dim=1)  # Shape: (B, H, W)

        h_clamped = torch.clamp(self.h, -0.05, 0.05)   # match energy_gradient()
        e_lin = -(s * h_clamped.view(1, -1, 1, 1)).sum(dim=1)

        return (e_quad + e_lin).sum(dim=[1, 2])  

    def energy_gradient(self, x):
        s = x[:, :self.chn, ...]
    
        # Enforce Hopfield symmetry & damped diagonal
        W_sym = self._get_constrained_W()
        h_clamped = torch.clamp(self.h, -0.05, 0.05)
        p_transformed = torch.einsum('ij,bjhw->bihw', W_sym, s) + h_clamped.view(1, -1, 1, 1)
        
        grad = p_transformed[:, :self.chn]
        return torch.clamp(grad, -1.0, 1.0)

    

    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        s_public = x[:, :self.chn, ...]
        energy_grad =  self.energy_gradient(x)
        update_mask = (torch.rand(b, 1, h, w, device=x.device) < update_rate)

        x_update = x + (y + self.eta * energy_grad) * update_mask * pre_life_mask

        # Bound hidden channels only, leave RGBA as-is
        v_part = x_update[:, :self.v_dim, ...]
        h_part = torch.tanh(x_update[:, self.v_dim:self.chn, ...])
        x_update = torch.cat([v_part, h_part], dim=1)

        post_life_mask = self.get_alive_mask(x_update)
        x_final = x_update * post_life_mask
        return x_final






        
        

