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



class EnergyOnlyNCA(nn.Module):
    def __init__(self, target_rgba, chn=16):
        super().__init__()
        self.chn = chn
        
        B, C, H, W = target_rgba.shape
        
        self.register_buffer("target_rgba", target_rgba)

        # W introduce a small noise on the hidden channels to make them growth 
        self.hidden_target = nn.Parameter(torch.randn(1, chn - 4, H, W) * 0.01)
        
        sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]) / 8.0
        self.register_buffer("Kx", sobel_x.view(1,1,3,3).repeat(chn,1,1,1))
        self.register_buffer("Ky", sobel_x.T.view(1,1,3,3).repeat(chn,1,1,1))
        
        # Learnable step size (eta)
        self.log_eta = nn.Parameter(torch.tensor(-3.0))

    def get_X_target(self):
        """ Dynamically combines RGBA target + learnable hidden prototypes """
        # Ensure hidden_target matches the batch size of target_rgba
        target_b = self.target_rgba.shape[0]
        hidden_b = self.hidden_target.shape[0]
    
        if hidden_b != target_b:
            # Expand hidden_target along batch dim to match target_rgba
            hidden = self.hidden_target.expand(target_b, -1, -1, -1)
        else:
            hidden = self.hidden_target

        return torch.cat([self.target_rgba, hidden], dim=1)

    def to_rgba(self, x):
        """ RGBA mapping (Direct slice since channels 0-3 explicitly target RGBA) """
        return x[:, :4, ...]

    def get_alive_mask(self, x):
        """ Check life based directly on Alpha (channel 3) """
        alpha = x[:, 3:4, :, :]
        padded = F.pad(alpha, [1, 1, 1, 1], mode="circular")
        return F.max_pool2d(padded, 3, stride=1, padding=0) > 0.1

    def perceive(self, s):
        s_padded = F.pad(s, [1,1,1,1], mode="circular")
        sx = F.conv2d(s_padded, self.Kx, groups=self.chn)
        sy = F.conv2d(s_padded, self.Ky, groups=self.chn)
        return torch.cat([s, sx, sy], dim=1)  # (B, 48, H, W)

    def energy(self, x):
        s = x[:, :self.chn, ...]
        p_s = self.perceive(s)
        p_X = self.perceive(self.get_X_target())
    
    # Calculate attractor loss ONLY on RGBA perception (first 3 * 4 = 12 channels of p)
    # This leaves hidden channels unconstrained by the target energy!
        p_s_rgba = torch.cat([p_s[:, :4], p_s[:, 16:20], p_s[:, 32:36]], dim=1)
        p_X_rgba = torch.cat([p_X[:, :4], p_X[:, 16:20], p_X[:, 32:36]], dim=1)
    
        return 0.5 * ((p_s_rgba - p_X_rgba) ** 2).sum(dim=[1, 2, 3]

    def energy_gradient(self, x):
        s = x[:, :self.chn, ...]
        p_s = self.perceive(s)
        p_X = self.perceive(self.get_X_target())
        
        diff = p_s - p_X
        d_id, d_sx, d_sy = diff[:, :self.chn], diff[:, self.chn:2*self.chn], diff[:, 2*self.chn:]
        
        s_padded_dsx = F.pad(d_sx, [1,1,1,1], mode="circular")
        s_padded_dsy = F.pad(d_sy, [1,1,1,1], mode="circular")
        
        # Adjoint perception step (transpose conv for Sobel)
        grad = d_id \
            - F.conv2d(s_padded_dsx, self.Kx, groups=self.chn) \
            - F.conv2d(s_padded_dsy, self.Ky, groups=self.chn)
            
        return torch.clamp(grad, -1.0, 1.0)

    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x).float()
        batch_n, chn_n, h, w = x.shape
        eta = F.softplus(self.log_eta)
        
        grad_E = self.energy_gradient(x)
        correction = torch.zeros_like(x)
        correction[:, :self.chn] = -eta * grad_E

        update_mask = (torch.rand(batch_n, 1, h, w, device=x.device) < update_rate).float()
        x_update = x + correction * update_mask * pre_life_mask
        
        post_life_mask = self.get_alive_mask(x_update).float()
        return x_update * post_life_mask



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







        
        

