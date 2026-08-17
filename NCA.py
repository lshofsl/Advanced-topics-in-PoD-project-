from multiprocessing import dummy
from sys import prefix

import torch 
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
        
        
        
class EnergyOnlyNCA(torch.nn.Module):
    def __init__(self, chn=16, v_dim=4):
        super().__init__()
        self.chn = chn
        self.v_dim = v_dim
        self.W = torch.nn.Parameter(torch.zeros(chn, chn))          # within-cell Hopfield term
        self.log_kappa = torch.nn.Parameter(torch.full((chn,), -15.0))       # per-channel diffusion strength, starting in 0.0
        self.log_eta = torch.nn.Parameter(torch.tensor(-2.0))

        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                          [1., -4., 1.],
                                          [0., 1., 0.]])
        self.register_buffer('lap_kernel', laplacian_kernel.view(1, 1, 3, 3))

    def get_alive_mask(self, x):
        alpha = x[:, 3:4, :, :]
        padded = torch.nn.functional.pad(alpha, [1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded, 3, stride=1, padding=0) > 0.1

    def _laplacian(self, s):
        s_padded = torch.nn.functional.pad(s, [1, 1, 1, 1], mode="circular")
        kernel = self.lap_kernel.repeat(self.chn, 1, 1, 1)     # depthwise, one Laplacian per channel
        return torch.nn.functional.conv2d(s_padded, kernel, groups=self.chn)

    def energy(self, x):
        # Total energy — BOTH terms, since this is what mono/drift logging must reflect
        s = x[:, :self.chn, ...]
        J_sym = (self.W + self.W.T) / 2
        Js = torch.einsum('nm,bmhw->bnhw', J_sym, s)
        E_hopfield = (-0.5 * (s * Js).sum(dim=1)).sum(dim=[1, 2])

        kappa = torch.nn.functional.softplus(self.log_kappa).view(1, -1, 1, 1)
        s_padded = torch.nn.functional.pad(s, [1, 1, 1, 1], mode="circular")
        # Dirichlet energy via unfold, or approximate directly from the Laplacian identity:
        # sum_i sum_j (s_i-s_j)^2 = -2 * sum_i s_i . laplacian(s)_i  (up to the deg*s_i^2 self term)
        lap = self._laplacian(s)
        E_spatial = (0.5 * kappa * (s * (-lap))).sum(dim=1).sum(dim=[1, 2])

        return E_hopfield + E_spatial

    def energy_gradient(self, x):
        s = x[:, :self.chn, ...]
        J_sym = (self.W + self.W.T) / 2
        Js = torch.einsum('nm,bmhw->bnhw', J_sym, s)
        grad_hopfield = -Js

        kappa = torch.nn.functional.softplus(self.log_kappa).view(1, -1, 1, 1)
        lap = self._laplacian(s)
        grad_spatial = -kappa * lap

        return grad_hopfield + grad_spatial
        
    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        b, c, h, w = x.shape
        eta = torch.nn.functional.softplus(self.log_eta)

        grad_E = self.energy_gradient(x)
        correction = torch.zeros_like(x)
        correction[:, :self.chn] = -eta * grad_E

        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()
        x_update = x + correction * update_mask   # x itself is preserved when a cell isn't updated

        v_part = x_update[:, :self.v_dim, ...]
        h_part = torch.tanh(x_update[:, self.v_dim:self.chn, ...])
        x_update = torch.cat([v_part, h_part], dim=1)

        post_life_mask = self.get_alive_mask(x_update)
        life_mask = (pre_life_mask & post_life_mask).float()   # AND, matching the reference implementation
        return x_update * life_mask



class HYBRID_NCA(torch.nn.Module):
    def __init__(self, chn=16, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3 * (chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.v_dim = 4
        self.h_dim = chn - self.v_dim
        self.W = torch.nn.Parameter(torch.randn(chn, chn) * 0.01)  # local within-cell coupling

    def get_alive_mask(self,x):
        alpha = x[:, 3:4, :, :] 
        padded_alpha = torch.nn.functional.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def energy(self, x):
        s_public = x[:, :self.chn, ...]
        J_sym = (self.W + self.W.T) / 2
        Js = torch.einsum('nm,bmhw->bnhw', J_sym, s_public)
        energy_density = -0.5 * (s_public * Js).sum(dim=1)  #Fix eta 
        return energy_density.sum(dim=[1, 2])
        
    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()

        x_update = x + y * update_mask * pre_life_mask

        v_part = x_update[:, :self.v_dim, ...]
        h_part = torch.tanh(x_update[:, self.v_dim:self.chn, ...])
        x_update = torch.cat([v_part, h_part], dim=1)

        post_life_mask = self.get_alive_mask(x_update)
        x_final = x_update * post_life_mask

        return x_final







        
        

