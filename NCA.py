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
    def __init__(self, v_dim=4, h_dim=9, u_dim=64, gene_size=3, sigma=0.5):
        super().__init__()
        self.v_dim, self.h_dim, self.u_dim = v_dim, h_dim, u_dim
        self.sigma = sigma
        self.chn = v_dim + h_dim          # Public channels (RGBA + hidden)
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
        
        # Low-rank gene modulation to drive the morphology landscape
        self.gene_proj = torch.nn.Conv2d(gene_size, v_dim * h_dim, 1, bias=False)
        torch.nn.init.normal_(self.gene_proj.weight, std=0.01)
        
    def compute_energy(self, v, h, b_eff, c_eff):
        """Calculates energy configuration per pixel."""
        v_term = ((v - b_eff) ** 2).sum(dim=1, keepdim=True) / (2 * self.sigma**2)
        Wv = self.W(v)                                    # (B, h_dim, H, W)
        wh_term = (Wv * h).sum(dim=1, keepdim=True) / self.sigma**2
        c_term = (c_eff * h).sum(dim=1, keepdim=True)
        return v_term - wh_term - c_term

    def forward(self, x, update_rate=0.5, settlement_steps=3):
        gene = x[:, -self.gene_size:, ...]  # Gene channels 
        s = x[:, :self.chn, ...]            # Public channels (RGBA + hidden)
        
        # 1. Compute Low-Rank Modulation Tensors
        delta_flat = self.gene_proj(gene)
        B_, _, H_, W_ = delta_flat.shape
        delta = delta_flat.view(B_, self.h_dim, self.v_dim, H_, W_) 

        # 2. Extract context via perception network
        u = reduced_perception(x[:, :(self.chn + self.gene_size)])
        
        b_eff = self.b + self.A(u)
        c_eff = self.c + self.B(u)
        
        # Initialize internal states for our ring settlement loop
        v_curr = x[:, :self.v_dim, ...].clone()
        h_curr = x[:, self.v_dim:self.chn, ...].clone()
        
        # 3. Cyclical Settlement Loop (Ring/Recurrent Information Exchange)
        for _ in range(settlement_steps):
            # Step A: Update Hidden channels using current Visible states
            v_exp = v_curr.unsqueeze(1)
            Wv_base = self.W(v_curr)
            Wv_gene = (delta * v_exp).sum(dim=2)
            h_curr = torch.sigmoid((Wv_base + Wv_gene) / self.sigma**2 + c_eff)
            
            # Step B: Update Visible channels using newly settled Hidden states
            hidden_exp = h_curr.unsqueeze(2)
            Wh_gene = (delta * hidden_exp).sum(dim=1) 
            v_curr = self.sigma**2 * (torch.nn.functional.conv_transpose2d(h_curr, self.W.weight) + Wh_gene
) + b_eff

        # Pack aggregated results back into public shapes
        s_new = torch.cat([v_curr, h_curr], dim=1)

        # 4. Standard Stochastic NCA Update Masking
        b, c, h, w = u.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()
        
        # Pad and pull the Alpha channel (Index 3) for the living mask
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0) > 0.1

        s_update = s + (s_new - s) * update_mask * pre_life_mask
        return torch.cat((s_update, gene), dim=1)
        
        

