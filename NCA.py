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
    
def get_alive_mask(self, x):

    alpha = x[:, 3:4, :, :] 
    padded_alpha = torch.nn.functional.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
    return torch.nn.functional.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

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
        
        
        
class NCA_EBM(torch.nn.Module):
    def __init__(self, chn=16, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3 * (chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()


        self.v_dim = 4
        self.h_dim = chn - self.v_dim
        self.J = torch.nn.Parameter(torch.zeros(chn, chn))  # local within-cell coupling
        self.eta = torch.nn.Parameter(torch.tensor(0.1))    # eta parameter
        #self.K = torch.nn.Parameter(torch.zeros(chn, chn))  # Interaction with neighbors

    def forward(self, x, update_rate=0.5):
        pre_life_mask = self.get_alive_mask(x)
        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        s_public = x[:, :self.chn, ...]
        J_sym = (self.J + self.J.T) / 2
        Js = torch.einsum('nm,bmhw->bnhw', J_sym, s_public)
        energy_grad = -Js
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()

        x_update = x + (y - self.eta * energy_grad) * update_mask * pre_life_mask

        # Bound hidden channels only, leave RGBA as-is
        v_part = x_update[:, :self.v_dim, ...]
        h_part = torch.tanh(x_update[:, self.v_dim:self.chn, ...])
        x_update = torch.cat([v_part, h_part], dim=1)

        post_life_mask = self.get_alive_mask(x_update)
        x_final = x_update * post_life_mask
        return x_final

    def energy(self, s):
        s_public = s[:, :self.chn, ...]
        J_sym = (self.J + self.J.T) / 2
        Js = torch.einsum('nm,bmhw->bnhw', J_sym, s_public)
        E = -0.5 * (s_public * Js).sum(dim=1, keepdim=True)
        return E








class RBM(torch.nn.Module):
    def __init__(self, v_dim=4, h_dim=9, gene_size=3):
        super().__init__()
        self.v_dim, self.h_dim = v_dim, h_dim
        self.gene_size = gene_size
        self.public = v_dim + h_dim 
        self.chn = v_dim + h_dim + gene_size

        # W is a 3x3 conv filter to get the interaction with the neighbors 
        self.W = torch.nn.Conv2d(v_dim, h_dim, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        torch.nn.init.normal_(self.W.weight, std=0.01)
        
        #Use of FiLM modulation
        self.film_v = torch.nn.Conv2d(64, v_dim, kernel_size=1)
        self.film_h = torch.nn.Conv2d(64, h_dim, kernel_size=1)

        self.a = torch.nn.Parameter(torch.zeros(1, v_dim, 1, 1))
        self.b = torch.nn.Parameter(torch.zeros(1, h_dim, 1, 1))
        self.log_sigma = torch.nn.Parameter(torch.zeros(1, v_dim, 1, 1))

    def forward(self, x, update_rate=0.5, eta = 0.1):
        gene = x[:, -self.gene_size:, ...]     # Gene channels 
        s = x[:, :self.public, ...]            # Public channels (RGBA + hidden)
        
        a_eff = self.a 
        b_eff = self.b 
        sigma = torch.exp(self.log_sigma)
        
        #We introduce the perception vector as a modulation in the weight matrix 
        y = reduced_perception(x[:, :self.chn], 0)
        
        gamma_v = torch.sigmoid(self.film_v(y)) * 2.0
        gamma_h = torch.sigmoid(self.film_h(y)) * 2.0
        
        
        v_curr = x[:, :self.v_dim, ...].clone()
        
        #Compute p(h|v)
        v_scaled = v_curr * gamma_v
        # Division of the visible state over sigma to scale correctly to the hidden states
        v_pre_conv = v_scaled / (sigma**2) 
        W_v = self.W(v_pre_conv)
        h_activation = (W_v * gamma_h) + b_eff
        h_curr = torch.sigmoid(h_activation)
            
        #Compute p(v|h) 
        h_scaled = h_curr * gamma_h
        w_t = self.W.weight.transpose(0, 1)
        w_t_flipped = torch.flip(w_t, dims=[2, 3])
        h_padded = F.pad(h_scaled, pad=[1, 1, 1, 1], mode="circular")
        W_t_h = F.conv2d(h_padded, w_t_flipped)
    
        v_curr = (W_t_h * gamma_v) * (sigma**2) + a_eff
        
        v_new, h_new = self.energy_gradient_step(v_curr, h_curr, gamma_v, gamma_h, eta)

        # Standard NCA Masking and Update
        s_new = torch.cat([v_new, h_new], dim=1)
        b, c, h, w = s.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + update_rate).floor()
        
        xmp = F.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = F.max_pool2d(xmp, 3, 1, 0) > 0.1

        s_update = s + (s_new - s) * update_mask * pre_life_mask
        return torch.cat((s_update, gene), dim=1)
        
        
        
        # FOR TRAINING 

    def sample_hidden(self, v, gamma_v, gamma_h):
        sigma = torch.exp(self.log_sigma)
        v_scaled = v * gamma_v  # FiLM scaling 
        
        # Divide by variance on the visible channels before convolution to prevent size mismatch
        v_pre_conv = v_scaled / (sigma**2)
        Wv = self.W(v_pre_conv)
        
        h_prob = torch.sigmoid((Wv * gamma_h) + self.b)
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample

    def sample_visible(self, h, gamma_v, gamma_h):
        sigma = torch.exp(self.log_sigma)
        h_scaled = h * gamma_h
        w_t = torch.flip(self.W.weight.transpose(0, 1), dims=[2, 3])
        h_padded = F.pad(h_scaled, pad=[1,1,1,1], mode="circular")
        W_t_h = F.conv2d(h_padded, w_t)
        
        # Bound the reconstructed visible states in [0, 1] range
        v_mean = torch.sigmoid(((W_t_h * gamma_v) * (sigma**2)) + self.a)
        return v_mean

    def compute_energy(self, v, h, gamma_v, gamma_h):
        sigma = torch.exp(self.log_sigma)  # Fixed NameError
        
        v_term = ((v - self.a)**2).sum(dim=1, keepdim=True) / (2 * sigma**2)
        
        # Divide by variance on the visible channels before convolution to prevent size mismatch
        v_pre_conv = (v * gamma_v) / (sigma**2)
        Wv = self.W(v_pre_conv)
        
        wh_term = ((Wv * gamma_h) * h).sum(dim=1, keepdim=True)
        c_term = (self.b * h).sum(dim=1, keepdim=True)
        return v_term - wh_term - c_term
        
    def gibbs_step(self, v, gamma_v, gamma_h):
        """One full v -> h -> v Gibbs sweep."""
        h_prob, h_sample = self.sample_hidden(v, gamma_v, gamma_h)
        v_new = self.sample_visible(h_sample, gamma_v, gamma_h)
        return v_new, h_prob, h_sample



## Energy-gradient descent of the RBM model 

    def energy_gradient_step(self, v, h, gamma_v, gamma_h, eta=0.1):

        v = v.detach().requires_grad_(True)
        h = h.detach().requires_grad_(True)
    
        E = self.compute_energy(v, h, gamma_v, gamma_h)  
        energy_sum = E.sum() 

        grad_v, grad_h = torch.autograd.grad(
            energy_sum, [v, h], create_graph=True)

        v_new = v - eta * grad_v
        h_new = h - eta * grad_h
        return v_new, h_new




    def contrastive_divergence(self, base, gene_data, k=1, eta=0.1, h_init=None):
        v_data = base.detach()
        B, _, H, W = v_data.shape

        # using a zero-hidden placeholder purely to build the perception vector as h is not on base image 
        h_placeholder = torch.zeros(B, self.h_dim, H, W, device=v_data.device)
        x_data = torch.cat([v_data, h_placeholder, gene_data], dim=1)
        y_data = reduced_perception(x_data, 0)
        gamma_v = torch.sigmoid(self.film_v(y_data)) * 2.0
        gamma_h = torch.sigmoid(self.film_h(y_data)) * 2.0

        h_prob_data, _ = self.sample_hidden(v_data, gamma_v, gamma_h)
        E_data = self.compute_energy(v_data, h_prob_data, gamma_v, gamma_h)

        # Negative phase: relax from the SAME target via energy-gradient descent,
        # k steps, matching the actual dynamics used in forward() -- not Gibbs
        # sampling, to keep this a fair test of the same mechanism.
        v_model, h_model = v_data.clone(), h_prob_data.clone()
        for _ in range(k):
            v_model, h_model = self.energy_gradient_step(v_model, h_model, gamma_v, gamma_h, eta)

        v_model = v_model.detach()
        h_model = h_model.detach()
        E_model = self.compute_energy(v_model, h_model, gamma_v, gamma_h)

        cd_loss = E_data.mean() - E_model.mean()
        return cd_loss


    def persistent_contrastive_divergence(self, v_target, v_actual, h_actual, gene_data, k=1):
        # --- 1. Positive Phase (using target image) ---
        # Construct perception vector for the target image (using zeros for target h)
        b, _, H, W = v_target.shape
        h_target_placeholder = torch.zeros(b, self.h_dim, H, W, device=v_target.device)
        x_target = torch.cat([v_target, h_target_placeholder, gene_data], dim=1)
    
        y_target = F.relu(reduced_perception(x_target, 0))
        gamma_v_t = torch.sigmoid(self.film_v(y_target)) * 2.0
        gamma_h_t = torch.sigmoid(self.film_h(y_target)) * 2.0
    
        # Sample hidden states that COHERENTLY match the target image
        h_prob_target, _ = self.sample_hidden(v_target, gamma_v_t, gamma_h_t)
        E_target = self.compute_energy(v_target, h_prob_target, gamma_v_t, gamma_h_t)

        # --- 2. Negative Phase (starting from the actual NCA states) ---
        # Construct perception vector for the actual noisy state
        x_actual = torch.cat([v_actual, h_actual, gene_data], dim=1)
        y_actual = F.relu(reduced_perception(x_actual, 0))
        gamma_v_a = torch.sigmoid(self.film_v(y_actual)) * 2.0
        gamma_h_a = torch.sigmoid(self.film_h(y_actual)) * 2.0

        # Start the Gibbs chain directly from the NCA's current state!
        v_model = v_actual.detach()
        for _ in range(k):
            v_model, h_prob_model, h_sample_model = self.gibbs_step(v_model, gamma_v_a, gamma_h_a)

        v_model = v_model.detach()
        h_prob_model = h_prob_model.detach()
        E_model = self.compute_energy(v_model, h_prob_model, gamma_v_a, gamma_h_a)
    
        return E_target.mean() - E_model.mean()














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
        
        

