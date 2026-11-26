"""Causal language models compared in the paper.

Every model maps token ids [B, T] to next-token logits [B, T, V] and sees only
tokens <= t when predicting token t+1.  All share the same embedding width d,
depth L, vocabulary and LM head, so differences come from the sequence-mixing
and feature-transformation blocks.

    mlp          fixed-window neural LM (Bengio et al., 2003)
    transformer  post-LN causal Transformer (Vaswani et al., 2017)
    rwkv         RWKV-4 time mixing + channel mixing (Peng et al., 2023)
    qrwkv        RWKV with a VQC branch in channel mixing (Fig. 3/4)
    qasa         Transformer whose last N layers insert a VQC before the FFN (Fig. 1/2)
    qksan        Transformer with fidelity-quantum-kernel attention and VQC value modulation

`<model>_c` (qrwkv_c, qasa_c, qksan_c) replaces every circuit by the classical
surrogate of the same input/output width (quantum.VQC(classical=True)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import quantum as Q
from .quantum import VQC, expval_z, reupload_state_grouped


@dataclass
class Config:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 64
    pad_id: int = 0
    # quantum
    n_qubits: int = 4
    q_depth: int = 2
    n_quantum_layers: int = 1  # QASA: how many of the n_layers are quantum encoder layers
    classical_surrogate: bool = False
    # mlp
    window: int = 4

    def to_dict(self):
        return asdict(self)


def causal_mask(T, device):
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


class LMBase(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def quantum_modules(self):
        return [m for m in self.modules() if isinstance(m, VQC) and not m.classical]


# ----------------------------------------------------------------------------
# MLP language model
# ----------------------------------------------------------------------------

class MLPLM(LMBase):
    """Concatenate embeddings of the last `window` tokens -> tanh hidden layer -> logits."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        k, d = cfg.window, cfg.d_model
        self.hidden = nn.Linear(k * d, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        B, T = ids.shape
        k = self.cfg.window
        padded = F.pad(ids, (k - 1, 0), value=self.cfg.pad_id)
        win = padded.unfold(1, k, 1)  # [B, T, k]  tokens t-k+1..t
        e = self.tok(win).reshape(B, T, -1)
        h = torch.tanh(self.hidden(self.drop(e)))
        return self.head(self.norm(self.proj(self.drop(h))))


# ----------------------------------------------------------------------------
# Transformer and QASA
# ----------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Post-LN layer: h = LN(x + MHA(x)); y = LN(h + FFN(h))."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, d))
        self.drop = nn.Dropout(cfg.dropout)

    def mix(self, x, mask, pad):
        a, _ = self.attn(x, x, x, attn_mask=mask, key_padding_mask=pad, need_weights=False)
        return self.ln1(x + self.drop(a))

    def forward(self, x, mask, pad):
        h = self.mix(x, mask, pad)
        return self.ln2(h + self.drop(self.ffn(h)))


class QuantumEncoderLayer(EncoderLayer):
    """QASA quantum encoder layer (Fig. 1, lower block).

    h  = LN(x + MHA(x))                                  attention sub-layer
    a  = LN_n(tanh(W_enc h + b_enc))     in R^n           (1) classical-to-quantum encoding (angles)
    z  = (<Z_1>, ..., <Z_n>) of U(theta) S(a)|0>         (2) VQC of Fig. 2 (n data qubits + 1 ancilla)
    u  = h + W_dec z + b_dec            in R^d            (3) measurement decoding back to width d
    y  = LN(h + FFN(u))
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        n = cfg.n_qubits
        self.enc = nn.Linear(cfg.d_model, n)
        self.enc_norm = nn.LayerNorm(n)
        self.vqc = VQC(n, cfg.q_depth, "qasa", classical=cfg.classical_surrogate)
        self.dec = nn.Linear(n, cfg.d_model)

    def forward(self, x, mask, pad):
        h = self.mix(x, mask, pad)
        a = self.enc_norm(torch.tanh(self.enc(h)))
        z = self.vqc(a)
        u = h + self.dec(z)
        return self.ln2(h + self.drop(self.ffn(u)))


class TransformerLM(LMBase):
    def __init__(self, cfg: Config, n_quantum: int = 0):
        super().__init__(cfg)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.ln_in = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layers = [EncoderLayer(cfg) for _ in range(cfg.n_layers - n_quantum)]
        layers += [QuantumEncoderLayer(cfg) for _ in range(n_quantum)]
        self.layers = nn.ModuleList(layers)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.ln_in(self.tok(ids) + self.pos(pos)[None]))
        mask = causal_mask(T, ids.device)
        for layer in self.layers:
            x = layer(x, mask, None)
        return self.head(x)


# ----------------------------------------------------------------------------
# QKSAN
# ----------------------------------------------------------------------------

class QuantumKernelAttention(nn.Module):
    """Multi-head attention with a fidelity quantum kernel and VQC value modulation.

    Per head a (d_h = d / h):
      S      = Q K^T / sqrt(d_h)
      psi_i  = |phi(q_i)>,  phi = data re-uploading circuit on n qubits, angles pi*tanh(P_a q_i)
      Gamma  = |<phi(q_i)|phi(k_j)>|^2          (Gram matrix of a PSD quantum kernel)
      A      = softmax_row(S + log(Gamma + eps) + M_causal)
      G_V    = sigmoid(U_a VQC_v(pi*tanh(R_a v_j)) + c_a);   V~ = V * G_V
      H_a    = A V~
    Classical surrogate: phi -> u = tanh(W pi*tanh(P_a q)) and Gamma_ij = exp(-||u_i - u_j||^2).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d, h, n, D = cfg.d_model, cfg.n_heads, cfg.n_qubits, cfg.q_depth
        self.h, self.dh, self.n = h, d // h, n
        self.classical = cfg.classical_surrogate
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.dropout)

        def lin(i, o):
            return nn.Parameter(torch.randn(h, i, o) / math.sqrt(i))

        self.feat_proj, self.val_proj = lin(self.dh, n), lin(self.dh, n)  # P_a, R_a
        self.val_up, self.val_up_b = lin(n, self.dh), nn.Parameter(torch.zeros(h, self.dh))  # U_a, c_a
        if self.classical:  # surrogate u = tanh(W a + b) for both the feature map and the value circuit
            self.feat_W, self.feat_b = lin(n, n), nn.Parameter(torch.zeros(h, n))
            self.val_W, self.val_b = lin(n, n), nn.Parameter(torch.zeros(h, n))
        else:
            self.feat_w = nn.Parameter(torch.rand(h, D, n, 3) * 2 * math.pi)
            self.val_w = nn.Parameter(torch.rand(h, D, n, 3) * 2 * math.pi)
        self.eps = 1e-6

    def circuit_params(self):
        return 0 if self.classical else self.feat_w.numel() + self.val_w.numel()

    def forward(self, x, mask):
        B, T, d = x.shape
        q, k, v = (t.reshape(B, T, self.h, self.dh) for t in self.qkv(x).chunk(3, dim=-1))
        S = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh)
        # angles for the feature map: same map phi_a for queries and keys -> symmetric PSD kernel
        aqk = math.pi * torch.tanh(torch.einsum("bthd,hdn->hbtn", torch.cat([q, k], 1), self.feat_proj))
        av = math.pi * torch.tanh(torch.einsum("bthd,hdn->hbtn", v, self.val_proj))
        if self.classical:
            u = torch.tanh(torch.einsum("hbtn,hnm->hbtm", aqk, self.feat_W) + self.feat_b[:, None, None])
            uq, uk = u[:, :, :T], u[:, :, T:]
            Gam = torch.exp(-torch.cdist(uq, uk) ** 2).permute(1, 0, 2, 3)
            z = torch.tanh(torch.einsum("hbtn,hnm->hbtm", av, self.val_W) + self.val_b[:, None, None])
        else:
            st = reupload_state_grouped(aqk.reshape(self.h, -1, self.n), self.feat_w)
            st = st.reshape(self.h, B, 2 * T, -1)
            ov = torch.einsum("hbik,hbjk->bhij", st[:, :, :T].conj(), st[:, :, T:])
            Gam = (ov.real ** 2 + ov.imag ** 2).float()
            if Q.SHOTS:  # finite-shot fidelity estimates (evaluation only)
                Gam = Q.shot_fidelity(Gam, Q.SHOTS)
            sv = reupload_state_grouped(av.reshape(self.h, -1, self.n), self.val_w)
            z = expval_z(sv.reshape(-1, sv.shape[-1]), self.n, self.n).reshape(self.h, B, T, self.n)
        A = torch.softmax(S + torch.log(Gam + self.eps) + mask[:, None], dim=-1)
        gate = torch.sigmoid(torch.einsum("hbtn,hnd->bthd", z, self.val_up) + self.val_up_b)
        vt = v * gate
        H = torch.einsum("bhij,bjhd->bihd", self.drop(A), vt)
        return self.out(H.reshape(B, T, d))


class QKSANLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.attn = QuantumKernelAttention(cfg)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        h = self.ln1(x + self.drop(self.attn(x, mask)))
        return self.ln2(h + self.drop(self.ffn(h)))


class QKSANLM(LMBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.ln_in = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([QKSANLayer(cfg) for _ in range(cfg.n_layers)])

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.ln_in(self.tok(ids) + self.pos(pos)[None]))
        mask = causal_mask(T, ids.device)[None]
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(x)


# ----------------------------------------------------------------------------
# RWKV and QRWKV
# ----------------------------------------------------------------------------

def token_shift(x):
    return F.pad(x, (0, 0, 1, -1))  # x_{t-1}, zeros at t=0


class TimeMixing(nn.Module):
    """RWKV-4 time mixing with the numerically stable WKV recurrence (state a, b, p)."""

    def __init__(self, cfg: Config, layer_id: int):
        super().__init__()
        d, L = cfg.d_model, cfg.n_layers
        r01 = layer_id / max(L - 1, 1)
        ddd = torch.arange(d).float() / d
        self.mu_k = nn.Parameter(ddd ** (1 - r01))
        self.mu_v = nn.Parameter(ddd ** (1 - r01) + 0.3 * (1 - r01))
        self.mu_r = nn.Parameter(ddd ** (0.5 * (1 - r01)))
        self.decay = nn.Parameter(-5 + 8 * (torch.arange(d).float() / max(d - 1, 1)) ** (0.7 + 1.3 * r01))
        self.first = nn.Parameter(torch.full((d,), math.log(0.3)))
        self.key, self.value = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.receptance, self.output = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        xx = token_shift(x)
        k = self.key(x * self.mu_k + xx * (1 - self.mu_k))
        v = self.value(x * self.mu_v + xx * (1 - self.mu_v))
        r = torch.sigmoid(self.receptance(x * self.mu_r + xx * (1 - self.mu_r)))
        w, u = -torch.exp(self.decay), self.first
        aa = torch.zeros(B, C, device=x.device)
        bb = torch.zeros(B, C, device=x.device)
        pp = torch.full((B, C), -1e38, device=x.device)
        outs = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            ww = u + kt
            p = torch.maximum(pp, ww)
            e1, e2 = torch.exp(pp - p), torch.exp(ww - p)
            outs.append((e1 * aa + e2 * vt) / (e1 * bb + e2))
            ww = pp + w
            p = torch.maximum(ww, kt)
            e1, e2 = torch.exp(ww - p), torch.exp(kt - p)
            aa, bb, pp = e1 * aa + e2 * vt, e1 * bb + e2, p
        return self.output(r * torch.stack(outs, dim=1))


class ChannelMixing(nn.Module):
    """RWKV channel mixing; with `quantum=True` a VQC branch is added (QRWKV, Fig. 3).

    xk, xr  token-shifted inputs;  r = sigmoid(W_r xr)
    c       = W_2 GELU(W_1 xk)                               classical FFN
    q       = W_dec VQC(W_enc xk)                            quantum branch (n qubits, Fig. 4)
    out     = r * W_f (c + q)          (RWKV baseline: out = r * W_f c)
    """

    def __init__(self, cfg: Config, layer_id: int, quantum: bool):
        super().__init__()
        d, L = cfg.d_model, cfg.n_layers
        r01 = layer_id / max(L - 1, 1)
        ddd = torch.arange(d).float() / d
        self.mu_k = nn.Parameter(ddd ** (1 - r01))
        self.mu_r = nn.Parameter(ddd ** (1 - r01))
        self.receptance = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, cfg.d_ff, bias=False), nn.Linear(cfg.d_ff, d, bias=False)
        self.fuse = nn.Linear(d, d, bias=False)
        self.quantum = quantum
        if quantum:
            self.enc = nn.Linear(d, cfg.n_qubits, bias=False)
            self.vqc = VQC(cfg.n_qubits, cfg.q_depth, "qrwkv", classical=cfg.classical_surrogate)
            self.dec = nn.Linear(cfg.n_qubits, d, bias=False)

    def forward(self, x):
        xx = token_shift(x)
        xk = x * self.mu_k + xx * (1 - self.mu_k)
        xr = x * self.mu_r + xx * (1 - self.mu_r)
        r = torch.sigmoid(self.receptance(xr))
        c = self.fc2(F.gelu(self.fc1(xk)))
        if self.quantum:
            c = c + self.dec(self.vqc(self.enc(xk)))
        return r * self.fuse(c)


class RWKVBlock(nn.Module):
    def __init__(self, cfg: Config, layer_id: int, quantum: bool):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.att = TimeMixing(cfg, layer_id)
        self.ffn = ChannelMixing(cfg, layer_id, quantum)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.att(self.ln1(x)))
        return x + self.drop(self.ffn(self.ln2(x)))


class RWKVLM(LMBase):
    def __init__(self, cfg: Config, quantum: bool):
        super().__init__(cfg)
        self.ln_in = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([RWKVBlock(cfg, i, quantum) for i in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model)

    def forward(self, ids):
        x = self.ln_in(self.tok(ids))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_out(x))


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------

# default circuit sizes follow the figures: QASA 8 data qubits + ancilla, 4 layers (Fig. 2);
# QRWKV 4 qubits, 2 entangling layers (Fig. 4); QKSAN 4 qubits per head, 2 re-uploading blocks.
QUANTUM_DEFAULTS = {"qasa": dict(n_qubits=8, q_depth=4), "qrwkv": dict(n_qubits=4, q_depth=2),
                    "qksan": dict(n_qubits=4, q_depth=2)}

MODEL_NAMES = ["mlp", "transformer", "rwkv", "qrwkv", "qrwkv_c", "qasa", "qasa_c", "qksan", "qksan_c"]


def build_model(name: str, cfg: Config) -> LMBase:
    base = name[:-2] if name.endswith("_c") else name
    if base in QUANTUM_DEFAULTS:
        cfg.classical_surrogate = name.endswith("_c")
    if name == "mlp":
        return MLPLM(cfg)
    if name == "transformer":
        return TransformerLM(cfg, n_quantum=0)
    if name == "rwkv":
        return RWKVLM(cfg, quantum=False)
    if base == "qrwkv":
        return RWKVLM(cfg, quantum=True)
    if base == "qasa":
        return TransformerLM(cfg, n_quantum=cfg.n_quantum_layers)
    if base == "qksan":
        return QKSANLM(cfg)
    raise ValueError(name)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    q = 0
    for m in model.modules():
        if isinstance(m, VQC) and not m.classical:
            q += sum(p.numel() for p in m.parameters())
    if isinstance(model, QKSANLM):
        q += sum(l.attn.circuit_params() for l in model.layers)
    emb = model.tok.weight.numel() + sum(p.numel() for p in model.head.parameters())
    return {"total": total, "circuit": q, "embedding_and_head": emb, "body": total - emb}
