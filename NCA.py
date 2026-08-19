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
        self.K_raw = torch.nn.Parameter(torch.randn(chn, chn, 3, 3) * 1e-2)
        self.log_eta = torch.nn.Parameter(torch.tensor(-4.0))
        self.log_a = torch.nn.Parameter(torch.full((chn,), -1.0))
        self.c = torch.nn.Parameter(torch.zeros(chn))
        self.log_b = torch.nn.Parameter(torch.full((chn,), -1.0))
        self.b = torch.nn.Parameter(torch.zeros(chn))

        self.log_gamma = torch.nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("K_hebb", torch.zeros(chn, chn, 3, 3))
        self.register_buffer("K_boundary", torch.zeros(chn, chn, 3, 3))
        self.log_gamma_boundary = torch.nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("K_sobel", torch.zeros(chn, chn, 3, 3))
        self.log_gamma_sobel = torch.nn.Parameter(torch.tensor(-1.0))

    def get_alive_mask(self, x):
        alpha = x[:, 3:4, :, :]
        padded = torch.nn.functional.pad(alpha, [1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded, 3, stride=1, padding=0) > 0.1

    def estimate_spectral_radius(self, num_iters=5, spatial_size=16):
        K = self._symmetric_kernel()
        eta = torch.nn.functional.softplus(self.log_eta)
        a = torch.nn.functional.softplus(self.log_a).view(1, -1, 1, 1)

        if not hasattr(self, "_pi_vec") or self._pi_vec.shape[1] != self.chn:
            v = torch.randn(1, self.chn, spatial_size, spatial_size, device=K.device)
            v = v / (v.norm() + 1e-8)
        else:
            v = self._pi_vec

        def apply_op(v):
            v_padded = torch.nn.functional.pad(v, [1, 1, 1, 1], mode="circular")
            Av = torch.nn.functional.conv2d(v_padded, K)
            return v + eta * Av + eta * a * v   # linearized forward-Euler step at s≈0, mask≈1

        for i in range(num_iters):
            Av = apply_op(v)
            v_new = Av / (Av.norm() + 1e-8)
            v = v_new if i == num_iters - 1 else v_new.detach()

        Av = apply_op(v)
        eigenvalue_est = Av.norm() / (v.norm() + 1e-8)

        self._pi_vec = v.detach()
        return eigenvalue_est

    def _dihedral_symmetrize(self, K):
        Kt = K.transpose(2, 3)  # reflect across the main diagonal
        transforms = [torch.rot90(base, k, dims=[2, 3]) for base in (K, Kt) for k in range(4)]
        return sum(transforms) / 8.0



    @torch.no_grad()
    def set_target_anchor(self, target):
        t = target[:, : self.v_dim, ...]
        live = (t[:, 3:4] > 0.1).float()
        n_live = live.sum().clamp(min=1.0)
        n_pixels = live.shape[-1] * live.shape[-2]

        offsets = [(-1, -1),(-1, 0),(-1, 1),
            (0, -1),(0, 0),(0, 1),(1, -1),(1, 0),(1, 1),]
        K_hebb = torch.zeros_like(self.K_raw)
        K_boundary = torch.zeros_like(self.K_raw)

        for dy, dx in offsets:
            shifted = torch.roll(t, shifts=(-dy, -dx), dims=(2, 3))
            shifted_live = torch.roll(live, shifts=(-dy, -dx), dims=(2, 3))

            corr = (
                torch.einsum("bnhw,bmhw,bohw->nm", t, shifted, live) / n_live
            )
            K_hebb[: self.v_dim, : self.v_dim, dy + 1, dx + 1] = corr

            corr_alpha = (live * shifted_live).sum() / n_pixels
            K_boundary[3, 3, dy + 1, dx + 1] = corr_alpha

        K_hebb_reflected = K_hebb.flip(dims=[2, 3]).transpose(0, 1)
        self.K_hebb.copy_(0.5 * (K_hebb + K_hebb_reflected))

        K_boundary_reflected = K_boundary.flip(dims=[2, 3]).transpose(0, 1)
        self.K_boundary.copy_(0.5 * (K_boundary + K_boundary_reflected))

        sobel_x = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                device=self.K_raw.device,
                dtype=self.K_raw.dtype,
            )
            / 8.0
        )
        sobel_y = sobel_x.T

        K_sobel = torch.zeros_like(self.K_raw)
        hidden_idx = self.v_dim

        for v in range(self.v_dim):
            if hidden_idx < self.chn:
                K_sobel[hidden_idx, v] = sobel_x  # Horizontal gradient
                hidden_idx += 1
            if hidden_idx < self.chn:
                K_sobel[hidden_idx, v] = sobel_y  # Vertical gradient
                hidden_idx += 1

        # Preserve spatial coupling symmetry
        K_sobel_reflected = K_sobel.flip(dims=[2, 3]).transpose(0, 1)
        self.K_sobel.copy_(0.5 * (K_sobel + K_sobel_reflected))

    def _symmetric_kernel(self):
        K_reflected = self.K_raw.flip(dims=[2, 3]).transpose(0, 1)
        K_learned_sym = 0.5 * (self.K_raw + K_reflected)   # existing self-adjointness (Hopfield) constraint
        K_learned_sym = self._dihedral_symmetrize(K_learned_sym)  # NEW: remove directional bias

        gamma = torch.nn.functional.softplus(self.log_gamma)
        gamma_boundary = torch.nn.functional.softplus(self.log_gamma_boundary)
        gamma_sobel = torch.nn.functional.softplus(self.log_gamma_sobel)

        return (
            K_learned_sym
            + gamma * self.K_hebb
            + gamma_boundary * self.K_boundary
            + gamma_sobel * self.K_sobel   # left untouched — directionality is the point
        )

    def _spatial_field(self, s, mask):
        s_masked = s * mask
        s_padded = torch.nn.functional.pad(
            s_masked, [1, 1, 1, 1], mode="circular"
        )
        K = self._symmetric_kernel()
        return torch.nn.functional.conv2d(s_padded, K)

    def energy(self, x):
        s = x[:, : self.chn, ...]
        mask = live_mask = self.get_alive_mask(x).to(x.dtype)
        
        h = self._spatial_field(s, mask)
        E_coupling = (-0.5 * (s * h).sum(dim=1)).sum(dim=[1, 2])

        b_bias = self.b.view(1, -1, 1, 1)
        E_bias = (b_bias * s).sum(dim=1).sum(dim=[1, 2])

        a = torch.nn.functional.softplus(self.log_a).view(1, -1, 1, 1)
        c = self.c.view(1, -1, 1, 1)
        E_reaction = (
            -0.5 * a * s.pow(2)
            + (1 / 3) * c * s.pow(3)).sum(dim=1).sum(dim=[1, 2])

        E_background_penalty = 0.001 * ((1.0 - mask) * s.pow(2)).sum(dim=[1, 2, 3])
        return E_coupling + E_bias + E_reaction + E_background_penalty


    
    def energy_gradient(self, x):
        s = x[:, : self.chn, ...]
        mask = self.get_alive_mask(x).to(x.dtype)

        h_masked = self._spatial_field(s, mask)                                      # A(Ms)
        s_padded = torch.nn.functional.pad(s, [1, 1, 1, 1], mode="circular")
        h_unmasked = torch.nn.functional.conv2d(s_padded, self._symmetric_kernel())  # A(s)
        grad_coupling = -0.5 * (h_masked + mask * h_unmasked)

        grad_bias = self.b.view(1, -1, 1, 1)
        a = torch.nn.functional.softplus(self.log_a).view(1, -1, 1, 1)
        c = self.c.view(1, -1, 1, 1)
        grad_reaction = -a * s + c * s.pow(2) 

        grad_background_penalty = 0.002 * (1.0 - mask) * s   # same mask, no cross-term needed

        total_grad = grad_coupling + grad_bias + grad_reaction + grad_background_penalty
        return total_grad

    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        batch_n, chn_n, h, w = x.shape
        eta = torch.nn.functional.softplus(self.log_eta)
        grad_E = self.energy_gradient(x)
        correction = torch.zeros_like(x)
        correction[:, : self.chn] = -eta * grad_E
        update_mask = (
            torch.rand(batch_n, 1, h, w, device=x.device) < update_rate
        )

        x_update = x + correction * update_mask * pre_life_mask
        post_life_mask = self.get_alive_mask(x_update)
        life_mask = (pre_life_mask & post_life_mask).float()
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







        
        

