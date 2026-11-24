""""Where is X now" box head for the stream: the image Glance box primitive (presence gate, patch softmax over the
latest frame group's visual tokens, 4 ordinal edge distributions, IoU confidence), read by one extra latent token
appended to the question block. BoxHeads / targets / losses follow the image Glance box heads (Vision-Decision/Glance-Core) so an image Glance box head
can initialise it (keys "box.*")."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BoxCfg:
    head_dim: int = 512
    edge_bins: int = 64
    edge_lo: float = -0.1
    edge_hi: float = 1.0
    edge_sigma_bins: float = 1.0
    w_presence: float = 1.0
    w_center: float = 1.0
    w_cramer: float = 0.1
    w_giou: float = 2.0
    w_iou_head: float = 0.5


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def fourier(xy, n=16):
    freqs = (2.0 ** torch.arange(n, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def box_iou_giou(a, b):
    x1 = torch.maximum(a[..., 0], b[..., 0]); y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2]); y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = lambda t: (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-9)
    cx1 = torch.minimum(a[..., 0], b[..., 0]); cy1 = torch.minimum(a[..., 1], b[..., 1])
    cx2 = torch.maximum(a[..., 2], b[..., 2]); cy2 = torch.maximum(a[..., 3], b[..., 3])
    c = ((cx2 - cx1) * (cy2 - cy1)).clamp(min=1e-9)
    return iou, iou - (c - union) / c


class BoxHeads(nn.Module):
    def __init__(self, D, cfg):
        super().__init__()
        self.cfg = cfg; d, K = cfg.head_dim, cfg.edge_bins; self.K = K
        self.q_norm = nn.LayerNorm(D); self.v_norm = nn.LayerNorm(D)
        self.presence = mlp(D, d, 1)
        self.c_q = nn.Linear(D, d); self.c_k = nn.Linear(D, d)
        self.edge = mlp(2 * D + 64, 2 * d, 4 * K)
        self.iou = mlp(2 * D + 64, d, 1)
        self.register_buffer("bin_centers", torch.linspace(cfg.edge_lo, cfg.edge_hi, K), persistent=False)

    def _feats(self, hq, v, pc):
        return torch.cat([hq, v, fourier(pc)], -1)

    def expected_box(self, el, pc):
        e = (el.softmax(-1) * self.bin_centers).sum(-1)
        return torch.stack([pc[:, 0] - e[:, 0], pc[:, 1] - e[:, 1], pc[:, 0] + e[:, 2], pc[:, 1] + e[:, 3]], -1)

    def forward(self, hq, vis, centers, sel_idx=None, *, reuse_features=False):
        """hq [M,D]; vis [M,N,D] visual tokens of the frame each question reads; centers [N,2]; sel_idx [M] (training)"""
        hq = self.q_norm(hq.float()); vis = self.v_norm(vis.float())
        out = dict(presence_logit=self.presence(hq).squeeze(-1))
        cl = torch.einsum("md,mnd->mn", self.c_q(hq), self.c_k(vis)) / math.sqrt(self.c_q.out_features)
        out["center_logits"] = cl
        pi = cl.argmax(-1); M = len(pi); ar = torch.arange(M, device=hq.device)
        vp, pcp = vis[ar, pi], centers[pi]
        features = self._feats(hq, vp, pcp)
        el = self.edge(features).view(M, 4, self.K)
        prediction = self.expected_box(el, pcp)
        iou_features = features if reuse_features else self._feats(hq, vp, pcp)
        out.update(box_pred=prediction, iou_logit=self.iou(iou_features).squeeze(-1), pred_idx=pi)
        if sel_idx is not None:
            vs, pcs = vis[ar, sel_idx], centers[sel_idx]
            els = self.edge(self._feats(hq, vs, pcs)).view(M, 4, self.K)
            out.update(edge_logits=els, pc_sel=pcs, box_sel=self.expected_box(els, pcs))
        return out


def patch_centers(gh, gw, device):
    ii, jj = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
    return torch.stack([(jj.flatten() + 0.5) / gw, (ii.flatten() + 0.5) / gh], -1)


def center_target_box(box, centers, gh, gw):
    """mass on every patch covered by the GT box, proportional to the covered area (the image Glance center target)"""
    hx, hy = 0.5 / gw, 0.5 / gh
    ox = (torch.minimum(centers[None, :, 0] + hx, box[:, 2:3]) - torch.maximum(centers[None, :, 0] - hx, box[:, 0:1])).clamp(min=0)
    oy = (torch.minimum(centers[None, :, 1] + hy, box[:, 3:4]) - torch.maximum(centers[None, :, 1] - hy, box[:, 1:2])).clamp(min=0)
    t = ox * oy
    return t / t.sum(-1, keepdim=True).clamp(min=1e-12)


def edge_targets(box, pc, cfg, K):
    offs = torch.stack([pc[:, 0] - box[:, 0], pc[:, 1] - box[:, 1], box[:, 2] - pc[:, 0], box[:, 3] - pc[:, 1]], -1)
    offs = offs.clamp(cfg.edge_lo, cfg.edge_hi)
    bw = (cfg.edge_hi - cfg.edge_lo) / (K - 1)
    c = torch.linspace(cfg.edge_lo, cfg.edge_hi, K, device=box.device)
    t = torch.exp(-0.5 * ((c - offs[..., None]) / (cfg.edge_sigma_bins * bw)) ** 2)
    return t / t.sum(-1, keepdim=True)


def cramer(logits, target):
    return ((logits.softmax(-1).cumsum(-1) - target.cumsum(-1)) ** 2).sum(-1).mean()


def box_loss(out, present, box, ct, cfg):
    L = {"presence": F.binary_cross_entropy_with_logits(out["presence_logit"], present.float())}
    pm = present.bool()
    if pm.any():
        L["center"] = -(ct[pm] * out["center_logits"][pm].log_softmax(-1)).sum(-1).mean()
        L["cramer"] = cramer(out["edge_logits"][pm], edge_targets(box[pm], out["pc_sel"][pm], cfg, out["edge_logits"].shape[-1]))
        L["giou"] = (1 - box_iou_giou(out["box_sel"][pm], box[pm])[1]).mean()
        with torch.no_grad():
            iou_pred = box_iou_giou(out["box_pred"][pm], box[pm])[0]
        L["iou_head"] = F.binary_cross_entropy_with_logits(out["iou_logit"][pm], (iou_pred >= 0.5).float())
    tot = cfg.w_presence * L["presence"] + sum(getattr(cfg, f"w_{k}") * v for k, v in L.items() if k != "presence")
    return tot, L


class BoxReadout(nn.Module):
    """one latent token after the question; its layer-`layer` hidden state is hq; vis = the same layer's hidden states of
    the latest frame group (Stream keeps them when st.vis_layer is set)."""

    def __init__(self, emb_weight, D=1024, cfg=None, layer=24):
        super().__init__()
        self.cfg = cfg or BoxCfg(); self.layer = layer
        mu, sd = emb_weight.float().mean(0), emb_weight.float().std(0)
        self.latent = nn.Parameter((mu + sd * torch.randn(1, D, device=emb_weight.device)).to(emb_weight.device))
        self.box = BoxHeads(D, self.cfg).to(emb_weight.device)

    def init_from_image(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["box"]
        missing, unexpected = self.box.load_state_dict(sd, strict=False)
        return missing, unexpected

    def forward(self, st, qids, rows=None, sel_idx=None, hires=None):
        """hires: optional (pair (1,2,3,H,W) in [0,1]) - B2: the current frame re-encoded at higher resolution, inserted
        into this question block only (never written to the stream); the box head then reads ITS tokens"""
        if not torch.is_grad_enabled() and hasattr(st, "read_box"):
            return st.read_box(self, qids, rows, sel_idx, hires)
        if hires is not None:
            return self._forward_hires(st, qids, rows, sel_idx, hires)
        N = len(qids); dev = st.dev
        lens = [len(x) for x in qids]; L = max(lens) + 1
        E = torch.zeros(N, L, self.latent.shape[1], device=dev, dtype=st.dtype)
        for i, x in enumerate(qids):
            E[i, : lens[i]] = st.emb(torch.tensor(x, device=dev))
        ar = torch.arange(N, device=dev); li = torch.tensor(lens, device=dev)
        E = E.index_put((ar, li), self.latent.to(st.dtype).expand(N, -1))
        cache = st.fork(N, rows)
        pos = torch.arange(st.pos, st.pos + L, device=dev).view(1, 1, -1).expand(3, N, -1)
        out = st.lm(inputs_embeds=E, position_ids=pos, past_key_values=cache, use_cache=True, output_hidden_states=True)
        hq = out.hidden_states[self.layer][ar, li]
        rws = torch.zeros(N, dtype=torch.long, device=dev) if rows is None else torch.as_tensor(rows, device=dev)
        vis = st.last_vis[rws]  # (N, n_tok, D)
        return self.apply_heads(hq, vis, st.last_grid, sel_idx)

    def apply_heads(self, hq, vis, grid, sel_idx):
        return self.box(hq, vis, patch_centers(*grid, hq.device), sel_idx)


    def hires_block(self, st, qids, pair):
        assert len(qids) == 1, "B2 readout: one question per call"
        dev = st.dev; ids = qids[0]
        vis, (gh, gw) = st.encode(pair)  # (1, gh*gw, D)
        vs, ve = st.emb.weight[st.ids["<|vision_start|>"]], st.emb.weight[st.ids["<|vision_end|>"]]
        # [vision_end of the stream][vision_start][hires tokens][vision_end] + question (+ latent); MRoPE like a frame group
        head = st.emb(torch.tensor(ids[:1], device=dev))  # the "<|vision_end|>" that closes the stream
        q = st.emb(torch.tensor(ids[1:], device=dev))
        E = torch.cat([head, vs[None], vis[0].to(st.dtype), ve[None], q, self.latent.to(st.dtype)], 0)[None]
        p = st.pos; P = [torch.full((3, 1), p, device=dev), torch.full((3, 1), p + 1, device=dev)]; p += 2
        r = torch.arange(gh, device=dev).repeat_interleave(gw); c = torch.arange(gw, device=dev).repeat(gh)
        P.append(torch.stack([torch.full_like(r, p), p + r, p + c])); p += max(gh, gw)
        n = len(ids) - 1 + 2
        P.append(torch.arange(p, p + n, device=dev).view(1, -1).expand(3, -1))
        P = torch.cat(P, 1)[:, None]
        return E, P, (gh, gw)

    def _forward_hires(self, st, qids, rows, sel_idx, pair):
        E, P, (gh, gw) = self.hires_block(st, qids, pair)
        cache = st.fork(1, rows)
        out = st.lm(inputs_embeds=E, position_ids=P, past_key_values=cache, use_cache=True, output_hidden_states=True)
        hs = out.hidden_states[self.layer][0]
        a0 = 2; hv = hs[a0:a0 + gh * gw][None]; hq = hs[-1:]
        return self.apply_heads(hq, hv, (gh, gw), sel_idx)
