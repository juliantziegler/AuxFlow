import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = ResidualBlock(96)
        self.resblock2 = ResidualBlock(96)

        self.proj = nn.Conv2d(96, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)   # ↓2
        x = self.layer2(x)   # ↓4
        x = self.resblock1(x)
        x = self.resblock2(x)
        return self.proj(x)


def flow_warp(x, flow):
    B, C, H, W = x.size()
    y, x_coord = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack((x_coord, y), dim=0).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    vgrid = grid + flow
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / (W - 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / (H - 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, align_corners=True)

class CorrelationLayer(nn.Module):
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_disp = max_displacement

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        corr = []
        for dy in range(-self.max_disp, self.max_disp + 1):
            for dx in range(-self.max_disp, self.max_disp + 1):
                shifted = torch.roll(f2, shifts=(dy, dx), dims=(2, 3))
                corr.append((f1 * shifted).sum(1, keepdim=True))
        return torch.cat(corr, dim=1)

class GlobalCorrelation(nn.Module):
    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        f1 = f1.view(B, C, H * W)
        f2 = f2.view(B, C, H * W)
        corr = torch.matmul(f1.transpose(1, 2), f2)  # [B, HW, HW]
        corr = corr / torch.sqrt(torch.tensor(C, dtype=torch.float32, device=f1.device))
        corr = corr.view(B, H, W, H * W).permute(0, 3, 1, 2)  # [B, H*W, H, W]
        return corr

class AttentionalCorrelation(nn.Module):
    def __init__(self, in_dim=128, attn_hidden_dim=64, out_dim=64):
        super().__init__()
        self.query_proj = nn.Conv2d(in_dim, attn_hidden_dim, 1)
        self.key_proj = nn.Conv2d(in_dim, attn_hidden_dim, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(attn_hidden_dim, 1, 1),  # scalar attention logits
            nn.Softmax(dim=-1)
        )
        self.out_dim = out_dim
        self.final_proj = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape

        # Project features
        query = self.query_proj(f1)  # [B, HIDDEN, H, W]
        key = self.key_proj(f2)      # [B, HIDDEN, H, W]

        # Flatten spatial
        query = query.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, HIDDEN]
        key = key.view(B, -1, H * W)                       # [B, HIDDEN, HW]

        # Compute dot product attention
        logits = torch.bmm(query, key)  # [B, HW, HW]
        attn_weights = F.softmax(logits, dim=-1)  # [B, HW, HW]

        # Use attention to compute weighted combination of original f2 features
        f2_flat = f2.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        attn_output = torch.bmm(attn_weights, f2_flat)  # [B, HW, C]

        # Reshape back to feature map
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        return self.final_proj(attn_output)  # reduce output dim (optional)




class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.convz = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h):
        hx = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h

class FlowUpdateBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gru = ConvGRU(input_dim, hidden_dim)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(self, x, h):
        h = self.gru(x, h)
        delta_flow = self.flow_head(h)
        return h, delta_flow

class FlowUpsampler(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2 * (scale ** 2), 1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, flow):
        return self.pixel_shuffle(self.conv(flow))

class SimpleRAFT(nn.Module):
    def __init__(self, iters=12, dummy_input_shape=(1, 3, 80, 80)):
        super().__init__()
        self.fnet = FeatureEncoder()
        self.iters = iters
        self.upsampler = FlowUpsampler(scale=4)
        self.corr = AttentionalCorrelation(in_dim=128, attn_hidden_dim=64, out_dim=64)

        # --- Dummy forward to infer input_dim ---
        with torch.no_grad():
            img1 = torch.randn(*dummy_input_shape)
            img2 = torch.randn(*dummy_input_shape)
            f1 = self.fnet(img1)
            f2 = self.fnet(img2)
            coords0, coords1 = self.initialize_flow(1, f1.shape[2], f1.shape[3])
            flow = coords1 - coords0
            f2_warped = flow_warp(f2, flow)
            corr_volume = self.corr(f1, f2_warped)
            x = torch.cat([f1, f2_warped, corr_volume, flow], dim=1)
            input_dim = x.shape[1]

        self.update_block = FlowUpdateBlock(input_dim=input_dim)

    def initialize_flow(self, B, Hf, Wf):
        coords0 = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing="ij")
        coords0 = torch.stack(coords0, dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1)
        coords1 = coords0.clone()
        return coords0, coords1


    def forward(self, img1, img2):
        B, C, H, W = img1.shape
        f1 = self.fnet(img1)
        f2 = self.fnet(img2)

        Hf, Wf = f1.shape[2], f1.shape[3]
        coords0, coords1 = self.initialize_flow(B, Hf, Wf)
        coords0, coords1 = coords0.to(img1.device), coords1.to(img1.device)

        h = torch.zeros(B, 128, Hf, Wf, device=img1.device)
        flow_preds = []

        for _ in range(self.iters):
            coords1 = coords1.detach()
            flow = coords1 - coords0
            f2_warped = flow_warp(f2, flow)
            corr_volume = self.corr(f1, f2_warped)  # [B, H*W, H, W]
            x = torch.cat([f1, f2_warped, corr_volume, flow], dim=1)  # [B, C, H, W]

            if self.update_block is None:
                self.update_block = FlowUpdateBlock(input_dim=x.shape[1])
                self.update_block.to(img1.device)

            h, delta_flow = self.update_block(x, h)
            coords1 = coords1 + delta_flow
            flow_preds.append(coords1 - coords0)


        flow_preds_up = [self.upsampler(f) for f in flow_preds]

        return flow_preds_up

# Example usage:
if __name__ == "__main__":
    model = SimpleRAFT()
    img1 = torch.randn(1, 3, 80, 80)
    img2 = torch.randn(1, 3, 80, 80)
    flow = model(img1, img2)
    print("Predicted flow shape:", flow[-1].shape)
    summary(model,)
