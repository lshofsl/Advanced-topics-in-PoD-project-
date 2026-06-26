from multiprocessing import dummy
from sys import prefix

import torch

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




class CRBM(torch.nn.Module):
    def __init__(self, v_dim=4, h_dim=9, u_dim=64, gene_size=3, sigma=1.0):
        super().__init__()
        self.v_dim, self.h_dim, self.u_dim = v_dim, h_dim, u_dim
        self.sigma = sigma
        self.chn = v_dim + h_dim          # public channels (RGBA + hidden)
        self.gene_size = gene_size

        # W as 1x1 conv: maps v_dim channels -> h_dim channels (per pixel)
        self.W = torch.nn.Conv2d(v_dim, h_dim, 1, bias=False)
        torch.nn.init.normal_(self.W.weight, std=0.01)

        self.b = torch.nn.Parameter(torch.zeros(1, v_dim, 1, 1))
        self.c = torch.nn.Parameter(torch.zeros(1, h_dim, 1, 1))

        # A, B as 1x1 convs: u_dim -> v_dim / h_dim, per pixel
        self.A = torch.nn.Conv2d(u_dim, v_dim, 1, bias=False)
        self.B = torch.nn.Conv2d(u_dim, h_dim, 1, bias=False)
        torch.nn.init.normal_(self.A.weight, std=0.01)
        torch.nn.init.normal_(self.B.weight, std=0.01)
        
        # To drive the morphologies in the energy landscape, we need to be projecting the genes stored in the weights, therefor they energy will not mixup during the training  
        #self.gene_bias_h = torch.nn.Conv2d(gene_size, h_dim, 1) -- The projection of the gene bias on the latent space help to improve the network but it needs a more strong 
        ##modulation, for this reason we move into a low-rank modulation 
        self.gene_proj = torch.nn.Conv2d(gene_size, v_dim * h_dim, 1, bias=False)
        torch.nn.init.normal_(self.gene_proj.weight, std=0.01)
        
        
    def compute_energy(self, v, h, b_eff, c_eff):
        v_term = ((v - b_eff) ** 2).sum(dim=1, keepdim=True) / (2 * self.sigma**2)
        Wh = torch.nn.functional.conv2d(v.unsqueeze(1), self.W.weight.unsqueeze(-1), padding=0)
        wh_term = (v / self.sigma**2 * Wh).sum(dim=1, keepdim=True)
        c_term = (c_eff * h).sum(dim=1, keepdim=True)
        E = v_term - wh_term - c_term
        return E

    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...] #Gene channels 
        s = x[:, :self.chn, ...]  # Public channels (hidden+RGBA)
        v = x[:, :self.v_dim, ...] #Visible channels (RBGA)
        
        delta_flat = self.gene_proj(gene)               # low-rank gene modulation 
        B_, _, H_, W_ = delta_flat.shape
        delta = delta_flat.view(B_, self.h_dim, self.v_dim, H_, W_) 

        # perception over RGBA+hidden+gene, gene only ever feeds u
        y = reduced_perception(x[:, :self.chn + self.gene_size], 0)
        u = y  
        
        v_exp = v.unsqueeze(1)
        Wv_base = self.W(v)
        Wv_gene = (delta * v_exp).sum(dim=2)

        b_eff = self.b + self.A(u)
        c_eff = self.c + self.B(u)

        # mean-field hidden (Bernoulli/sigmoid), conv W acts as v->h map
        hidden = torch.sigmoid((Wv_base + Wv_gene) / (self.sigma**2) + c_eff)
        hidden_exp = hidden.unsqueeze(2)                      # (B, h_dim, 1, H, W)
        Wh_gene = (delta * hidden_exp).sum(dim=1) 

        # mean-field visible reconstruction (Gaussian), W^T via conv_transpose
        v_new = torch.nn.functional.conv_transpose2d(hidden, self.W.weight) + Wh_gene + b_eff

        s_new = torch.cat([v_new, hidden], dim=1)

        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0).cuda() > 0.1

        s_update = s + (s_new - s) * update_mask * pre_life_mask
        x = torch.cat((s_update, gene), dim=1)
        
        
        #Energy 
        E = self.compute_energy(v_new, hidden, b_eff, c_eff)
        
        return x
        
        
#To have a learnable, no explicit energy function, we can work with the same small MLP as the baseline NCA.         

class Energy_learnable(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn #private channels
        self.w1 = torch.nn.Conv2d(chn + 3 * (chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn - gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def compute_energy(self, x):
        """
        Compute total energy E_θ(x) = sum_i e_θ(P_i(x))
        x: (B, C, H, W), requires_grad=True
        Returns: scalar energy (accumulated over batch and grid)
        """
        # Get perception for all cells
        y = reduced_perception(x[:, :self.chn], 0)
        
        # Evaluate energy density at each cell via MLP
        energy_density = torch.relu(self.w1(y))  # (B, hidden_n, H, W)
        energy_density = self.w2(energy_density)  # (B, 1, H, W)
        
        # Sum over all cells to get total energy
        energy = energy_density.sum()
        
        return energy
    
    def forward(self, x, steps=32, eta=0.01, update_rate=0.5, return_trajectory=False):
        """
        Energy-gradient NCA update
        
        Args:
            x: (B, C, H, W) state
            steps: number of gradient descent steps
            eta: step size for gradient descent
            update_rate: probability of updating each cell (standard NCA)
            return_trajectory: if True, return trajectory; else just final state
        
        Returns:
            x_out: (B, C, H, W) updated state
            (optional) trajectory, energies if return_trajectory=True
        """
        B, C, H, W = x.shape
        
        # Separate gene channels (static, don't update)
        gene = x[:, -self.gene_size:, ...]
        x_dynamic = x[:, :-self.gene_size, ...]
        
        trajectory = [x.detach().clone()]
        energies = []
        
        # Gradient descent on energy
        x_current = x.clone().requires_grad_(True)
        
        for t in range(steps):
            # Compute energy
            energy = self.compute_energy(x_current)
            energies.append(energy.item())
            
            # Compute gradient ∇_x E
            grad_x = torch.autograd.grad(
                energy, x_current,
                create_graph=False,
                retain_graph=False)[0]
            
            # Update: x ← x - η ∇_x E
            with torch.no_grad():
                x_new = x_current - eta * grad_x
                trajectory.append(x_new.clone())
            
            # Reattach for next iteration
            x_current = x_new.detach().requires_grad_(True)
        
        x_out = x_current.detach()

        update_mask = (torch.rand(B, 1, H, W, device=x.device) + update_rate).floor()
        xmp = torch.nn.functional.pad(x_out[:, 3:4, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0) > 0.1
        x_out = x_out * update_mask * pre_life_mask
        # Restore gene channels
        x_out = torch.cat((x_out[:, :-self.gene_size, ...], gene), dim=1)
        
        if return_trajectory:
            return x_out, torch.tensor(energies, device=x.device), trajectory
        else:
            return x_out
        

