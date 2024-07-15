import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

from hash_encoding import HashEmbedder, SHEncoder,HashEmbedder_2d

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # original raw input "x" is also included in the output
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, scalar_factor=1,PE = 1,log_map = 8,max_re_v = 512,level = 8):
    
    min_bound = [0, 0]
    max_bound = [1, 1 ]

    
    bounding_box = (torch.tensor(min_bound).cuda(), torch.tensor(max_bound).cuda())
    
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    if PE == 0:
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x/scalar_factor)
        out_dim = embedder_obj.out_dim
    elif PE == 1:
        embed = HashEmbedder_2d(bounding_box = bounding_box, base_resolution=16, finest_resolution=max_re_v,n_levels = level)
                            
        out_dim = embed.out_dim
    elif PE == 2:
        embed = HashEmbedder_2d(bounding_box = bounding_box,log2_hashmap_size=log_map, base_resolution=16, finest_resolution=max_re_v,n_levels = level)
                            
        out_dim = embed.out_dim
    return embed, out_dim


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

class NeRO(nn.Module):

    def __init__(self, num_semantic_classes, D=8, W=256, input_ch=3,skips=[4],max_val = None,min_val = None,mod = None,sem = True):
        super(NeRO, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.max_val = max_val
        self.min_val = min_val
        self.mod = mod
        self.sem = sem
        
        if self.mod == 1:
            self.embed_fn, self.input_ch = get_embedder(10, 0, scalar_factor=1,PE = 1,max_re_v = 4096,level = 16)
            self.embed_fn_z, self.input_ch_z = get_embedder(10, 0, scalar_factor=1,PE = 2,log_map = 8,max_re_v = 512,level = 8)
            self.embed_fn_s, self.input_ch_s = get_embedder(10, 0, scalar_factor=1,PE = 2,log_map = 24,max_re_v = 512,level = 8)
        else:
            self.embed_fn, self.input_ch = get_embedder(10, 0, scalar_factor=1,PE = 0)
            self.embed_fn_z, self.input_ch_z = get_embedder(10, 0, scalar_factor=1,PE = 0)
            self.embed_fn_s, self.input_ch_s = get_embedder(10, 0, scalar_factor=1,PE = 0)
        
        self.skips = skips

        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])


        self.views_linears = nn.ModuleList([nn.Linear(W, W)])

        input_ch_s = self.input_ch_s

        self.semantic_linear = nn.Sequential(fc_block(input_ch_s, W // 2), nn.Linear(W // 2, num_semantic_classes))

        input_ch_z = self.input_ch_z

        self.z_linears = nn.ModuleList([nn.Linear(input_ch_z, W)] + [nn.Linear(W, W)])
        self.z_linear = nn.Linear(W, 1)
        rgb_size = 1
        self.rgb_linears = nn.ModuleList([nn.Linear(W, W) for i in range(rgb_size)])
        self.rgb_linear = nn.Linear(W, 3)


        

    def forward(self, x):

        sem_in = x
        
        mean_val = (torch.tensor(self.min_val).float().cuda() + torch.tensor(self.max_val).float().cuda()) / 2
        x = ((x - torch.tensor(self.min_val).float().cuda()) / (torch.tensor(self.max_val).float().cuda() - torch.tensor(self.min_val).float().cuda()))
        input_pts = self.embed_fn(x)
        if self.sem:
            input_sem = self.embed_fn_s(x)
        z_out = self.embed_fn_z(x)
        h = input_pts

        for i, l in enumerate(self.z_linears):
            
            z_out = self.z_linears[i](z_out)
            z_out = F.relu(z_out)
        z = self.z_linear(z_out)
        for i, l in enumerate(self.pts_linears):

            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.sem:

            sem_logits = self.semantic_linear(input_sem)


        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        for i, l in enumerate(self.rgb_linears):
            h = self.rgb_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)
        if self.sem:
            outputs = torch.cat([z,rgb,sem_logits], -1)
        else:
            outputs = torch.cat([z,rgb], -1)
        return outputs
