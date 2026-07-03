Nca merged corrected · PY
from typing import Literal, Optional
import torch
import torch.nn as nn                       
import torch.nn.functional as F
 

 
def make_filters(device: torch.device) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device)
    lap = torch.tensor(
        [[1., 2., 1.], [2., -12., 2.], [1., 2., 1.]], device=device)
    return torch.stack([sobel_x, sobel_x.T, lap])   # (3, 3, 3)
 
 

def perchannel_conv(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = F.pad(y, [1, 1, 1, 1], 'circular')
    y = F.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)
 
 
def perception(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Full perception: [state | sobel_x | sobel_y | laplacian]  (4 × C channels)."""
    obs = perchannel_conv(x, filters)
    return torch.cat([x, obs], dim=1)           # (B, 4*C, H, W)
 
 
def reduced_perception(
    x: torch.Tensor,
    filters: torch.Tensor,
    mask_n: int = 0,
) -> torch.Tensor:
    """Perception on first (C - mask_n) channels, full state concatenated."""
    x_redu = x[:, :x.shape[1] - mask_n]
    obs = perchannel_conv(x_redu, filters)
    return torch.cat([x, obs], dim=1)
 

 
class SplitHeadMLP(nn.Module):
    """
    Two independent MLP heads, each receiving a controlled slice of the
    perception vector.
    """
 
    def __init__(
        self,
        perc_dim:   int,
        n_visible:  int,
        n_hidden:   int,
        hidden_dim: int = 64,
        mode:       Literal['full', 'cross_only', 'self_only'] = 'full',
        n_filters:  int = 3,          # sobel_x, sobel_y, laplacian
    ):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.n_ch      = n_visible + n_hidden
        self.mode      = mode
 
        # ── Build index masks ──────────────────────────────────────────────
        # Perception layout (n_groups = n_filters + 1 for identity):
        #   group 0 : identity  (offsets 0          …  n_ch-1)
        #   group 1 : sobel_x   (offsets n_ch        …  2*n_ch-1)
        #   group 2 : sobel_y   (offsets 2*n_ch       …  3*n_ch-1)
        #   group 3 : laplacian (offsets 3*n_ch       …  4*n_ch-1)
        # Within each group: first n_visible = visible, next n_hidden = hidden.
        #
        # Bug 2 fix: original code had [0, n_ch, 3*n_ch] — missing 2*n_ch.
        n_groups = n_filters + 1        # identity + 3 filter outputs
        n_ch = self.n_ch
        vis_idx, hid_idx = [], []
        for g in range(n_groups):
            offset = g * n_ch
            vis_idx += list(range(offset,             offset + n_visible))
            hid_idx += list(range(offset + n_visible, offset + n_ch))
 
        self.register_buffer('vis_idx', torch.tensor(vis_idx))
        self.register_buffer('hid_idx', torch.tensor(hid_idx))
 
        n_vis_feats = len(vis_idx)   # n_visible * n_groups
        n_hid_feats = len(hid_idx)   # n_hidden  * n_groups
 
        # ── Input dims per mode ────────────────────────────────────────────
        if mode == 'full':
            vis_in = hid_in = perc_dim
        elif mode == 'cross_only':
            vis_in = n_hid_feats   # visible output ← hidden perception
            hid_in = n_vis_feats   # hidden  output ← visible perception
        elif mode == 'self_only':
            vis_in = n_vis_feats   # visible output ← visible perception
            hid_in = n_hid_feats   # hidden  output ← hidden  perception
        else:
            raise ValueError(f"Unknown mode '{mode}'")
 
        # ── Heads ──────────────────────────────────────────────────────────
        self.vis_head = nn.Sequential(
            nn.Linear(vis_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_visible),
        )
        self.hid_head = nn.Sequential(
            nn.Linear(hid_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_hidden),
        )
 
        nn.init.zeros_(self.vis_head[-1].weight)
        nn.init.zeros_(self.vis_head[-1].bias)
        nn.init.zeros_(self.hid_head[-1].weight)
        nn.init.zeros_(self.hid_head[-1].bias)
 
    def _select(self, flat: torch.Tensor, which: str) -> torch.Tensor:
        if which == 'all':
            return flat
        idx = self.vis_idx if which == 'vis' else self.hid_idx
        return flat[:, idx]
 
    def forward(self, perc: torch.Tensor) -> torch.Tensor:
        """
        perc : (B, 4*C, H, W)   full perception tensor from perception()
        returns : (B, n_visible + n_hidden, H, W)   update delta
        """
        B, _, H, W = perc.shape
        flat = perc.permute(0, 2, 3, 1).reshape(-1, perc.shape[1])  # (B*H*W, 4C)
 
        if self.mode == 'full':
            vis_delta = self.vis_head(self._select(flat, 'all'))
            hid_delta = self.hid_head(self._select(flat, 'all'))
        elif self.mode == 'cross_only':
            vis_delta = self.vis_head(self._select(flat, 'hid'))
            hid_delta = self.hid_head(self._select(flat, 'vis'))
        elif self.mode == 'self_only':
            vis_delta = self.vis_head(self._select(flat, 'vis'))
            hid_delta = self.hid_head(self._select(flat, 'hid'))
 
        delta = torch.cat([vis_delta, hid_delta], dim=-1)   # (B*H*W, n_ch)
        return delta.reshape(B, H, W, self.n_ch).permute(0, 3, 1, 2)
 
 

 
class GeneCA(nn.Module):
    """
    GeneCA baseline.
 
    Channel layout:  [RGBA (0:4) | hidden (4:chn-gene_size) | gene (chn-gene_size:chn)]
 
    Parameters
    ----------
    chn           : total number of channels (default 12)
    hidden_n      : MLP hidden width
    gene_size     : number of frozen gene channels at the end (default 3)
    pathway_mode  : None (standard MLP) | 'full' | 'cross_only' | 'self_only'
                    When not None, replaces the standard w1/w2 with SplitHeadMLP.
    """
 
    def __init__(
        self,
        chn:          int  = 12,
        hidden_n:     int  = 96,
        gene_size:    int  = 3,
        pathway_mode: Optional[Literal['full', 'cross_only', 'self_only']] = None,
        device:       str  = 'cuda:0',
    ):
        super().__init__()
        self.chn       = chn
        self.gene_size = gene_size
        self.device    = device
 
        n_rgba    = 4
        n_hidden  = chn - gene_size - n_rgba   # non-gene, non-RGBA channels
        n_updated = chn - gene_size             # channels that receive updates
 
        # Perception: 3 filters + identity → 4 groups of chn channels each
        perc_dim = 4 * chn
        self.register_buffer('filters', make_filters(torch.device(device)))
 
        self.pathway_mode = pathway_mode
 
        if pathway_mode is None:
            # ── Standard GeneCA (original architecture) ────────────────────
            self.w1 = nn.Conv2d(perc_dim, hidden_n, 1)
            self.w2 = nn.Conv2d(hidden_n, n_updated, 1, bias=False)
            nn.init.zeros_(self.w2.weight)
        else:
            self.update_mlp = SplitHeadMLP(
                perc_dim   = perc_dim,
                n_visible  = n_rgba,
                n_hidden   = n_hidden,
                hidden_dim = hidden_n,
                mode       = pathway_mode,
                n_filters  = 3,
            )
 
    def forward(self, x: torch.Tensor, update_rate: float = 0.5) -> torch.Tensor:
        gene = x[:, -self.gene_size:]                
        perc = perception(x, self.filters)           
 
        if self.pathway_mode is None:
            y = self.w2(torch.relu(self.w1(perc)))   
        else:
            y = self.update_mlp(perc)                
 
        # Stochastic Update
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()
 
        # Living mask
        xmp = F.pad(x[:, None, 3], pad=[1, 1, 1, 1], mode='circular')
        pre_life_mask = F.max_pool2d(xmp, 3, 1, 0).to(self.device) > 0.1
 
        # Public channels update
        x_updated = x[:, :x.shape[1] - self.gene_size] + y * update_mask * pre_life_mask
        return torch.cat([x_updated, gene], dim=1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL HELPER VARIANTS (kept as-is, just device-agnostic filters)
# ─────────────────────────────────────────────────────────────────────────────
 
class DummyVCA(nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0, device='cuda:0'):
        super().__init__()
        self.chn    = chn
        self.mask_n = mask_n
        self.device = device
        self.register_buffer('filters', make_filters(torch.device(device)))
        self.w1 = nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = nn.Conv2d(hidden_n, chn, 1, bias=False)
        nn.init.zeros_(self.w2.weight)
 
    def forward(self, x, update_rate=0.5):
        y = perception(x, self.filters)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()
        xmp = F.pad(x[:, None, 3], pad=[1, 1, 1, 1], mode='circular')
        pre_life_mask = F.max_pool2d(xmp, 3, 1, 0).to(self.device) > 0.1
        return x + y * update_mask * pre_life_mask
 
 
class ReducedCA(nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0, device='cuda:0'):
        super().__init__()
        self.chn    = chn
        self.mask_n = mask_n
        self.device = device
        self.register_buffer('filters', make_filters(torch.device(device)))
        self.w1 = nn.Conv2d(chn + 3 * (chn - mask_n), hidden_n, 1)
        self.w2 = nn.Conv2d(hidden_n, chn, 1, bias=False)
        nn.init.zeros_(self.w2.weight)
 
    def forward(self, x, update_rate=0.5):
        y = reduced_perception(x, self.filters, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()
        xmp = F.pad(x[:, None, 3], pad=[1, 1, 1, 1], mode='circular')
        pre_life_mask = F.max_pool2d(xmp, 3, 1, 0).to(self.device) > 0.1
        return x + y * update_mask * pre_life_mask
 
 
# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    device = 'cpu'
    B, C, H, W = 2, 12, 32, 32
 
    for mode in [None, 'full', 'cross_only', 'self_only']:
        model = GeneCA(chn=C, hidden_n=64, gene_size=3,
                       pathway_mode=mode, device=device)
        x = torch.zeros(B, C, H, W)
        x[:, 3, H//2, W//2] = 1.0   # alpha seed
        out = model(x)
        label = mode if mode else 'standard'
        assert out.shape == (B, C, H, W), f"Shape mismatch for mode={label}"
        print(f"[{label:12s}]  output shape: {tuple(out.shape)}  ✓")

        
        




