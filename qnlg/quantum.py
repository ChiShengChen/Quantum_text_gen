"""Batched state-vector simulation of the variational quantum circuits (VQCs).

All circuits are simulated exactly (no shot noise) as complex state vectors in
PyTorch, so that they run batched on GPU and are differentiable by ordinary
back-propagation.  Wire 0 is the most significant qubit, matching the PennyLane
convention; `tests/test_quantum.py` checks every circuit against PennyLane.

Three circuit families are used, one per hybrid model:

* `qasa_circuit`   - the QASA circuit of Fig. 2 (n data qubits + 1 ancilla).
* `qrwkv_circuit`  - angle embedding + BasicEntanglerLayers (Fig. 4).
* `reupload_state` - data re-uploading feature-map state used by QKSAN's
                     fidelity kernel and value modulation.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn

CDTYPE = torch.complex64


# ----------------------------------------------------------------------------
# Elementary operations on a batch of states psi: [B, 2**n] (complex)
# ----------------------------------------------------------------------------

def zero_state(batch: int, n: int, device) -> torch.Tensor:
    psi = torch.zeros(batch, 2 ** n, dtype=CDTYPE, device=device)
    psi[:, 0] = 1.0
    return psi


def _apply_1q(psi: torch.Tensor, U: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    """Apply a single-qubit gate. U is [2,2] (shared) or [B,2,2] (per sample)."""
    B = psi.shape[0]
    a, c = 2 ** wire, 2 ** (n - wire - 1)
    view = psi.reshape(B, a, 2, c)
    if U.dim() == 2:
        out = torch.einsum("ij,bajc->baic", U, view)
    else:
        out = torch.einsum("bij,bajc->baic", U, view)
    return out.reshape(B, 2 ** n)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    c, s = c.to(CDTYPE), s.to(CDTYPE)
    m = torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)
    return m


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2).to(CDTYPE), torch.sin(theta / 2).to(CDTYPE)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    e_m = torch.exp(-0.5j * theta.to(CDTYPE))
    e_p = torch.exp(0.5j * theta.to(CDTYPE))
    z = torch.zeros_like(e_m)
    return torch.stack([torch.stack([e_m, z], -1), torch.stack([z, e_p], -1)], -2)


def rx(psi, theta, wire, n):
    return _apply_1q(psi, _rx(theta), wire, n)


def ry(psi, theta, wire, n):
    return _apply_1q(psi, _ry(theta), wire, n)


def rz(psi, theta, wire, n):
    return _apply_1q(psi, _rz(theta), wire, n)


@lru_cache(maxsize=None)
def _cnot_perm(control: int, target: int, n: int) -> tuple:
    perm = []
    for k in range(2 ** n):
        if (k >> (n - 1 - control)) & 1:
            perm.append(k ^ (1 << (n - 1 - target)))
        else:
            perm.append(k)
    return tuple(perm)


_PERM_CACHE: dict = {}


def cnot(psi: torch.Tensor, control: int, target: int, n: int) -> torch.Tensor:
    key = (control, target, n, psi.device)
    idx = _PERM_CACHE.get(key)
    if idx is None:
        idx = torch.tensor(_cnot_perm(control, target, n), dtype=torch.long, device=psi.device)
        _PERM_CACHE[key] = idx
    return psi.index_select(1, idx)


SHOTS = None  # None: exact expectation values; int S: finite-shot estimates (evaluation only)


def sample_probs(probs: torch.Tensor, shots: int) -> torch.Tensor:
    """Empirical distribution of `shots` computational-basis measurements per row."""
    rows, K = probs.shape
    out = torch.empty_like(probs)
    chunk = max(1, int(2e7 // shots))
    for s in range(0, rows, chunk):
        p = probs[s:s + chunk].clamp_min(0)
        idx = torch.multinomial(p / p.sum(-1, keepdim=True), shots, replacement=True)
        cnt = torch.zeros_like(p).scatter_add_(1, idx, torch.ones_like(idx, dtype=p.dtype))
        out[s:s + chunk] = cnt / shots
    return out


def shot_fidelity(fid: torch.Tensor, shots: int) -> torch.Tensor:
    """Finite-shot estimate of a fidelity measured by a compute-uncompute test:
    the fraction of `shots` runs that return the all-zero bitstring, Binomial(S, F)/S."""
    return torch.distributions.Binomial(shots, probs=fid.clamp(0, 1)).sample() / shots


def expval_z(psi: torch.Tensor, n_measure: int, n: int) -> torch.Tensor:
    """<Z_i> for wires 0..n_measure-1. Returns real [B, n_measure].

    With SHOTS set, all Z_i are estimated from the same SHOTS sampled bitstrings,
    as on hardware (joint sampling, so correlations between qubits are kept)."""
    B = psi.shape[0]
    probs = psi.real ** 2 + psi.imag ** 2
    if SHOTS:
        probs = sample_probs(probs, SHOTS)
    outs = []
    for w in range(n_measure):
        v = probs.reshape(B, 2 ** w, 2, 2 ** (n - w - 1)).sum(dim=(1, 3))
        outs.append(v[:, 0] - v[:, 1])
    return torch.stack(outs, dim=-1)


# ----------------------------------------------------------------------------
# Circuit families
# ----------------------------------------------------------------------------

def qasa_circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """QASA VQC (Fig. 2): x [B, n] angles, weights [L, n+1]; returns <Z> [B, n].

    Layer 0: RX(x_i) RZ(x_i) encoding, then trainable RX(w0_i) RZ(w1_i).
    Layers l=1..L-1: ring of CNOT(i, i+1 mod n) each followed by RY(w_l,i) RZ(w_l,i)
    on the data qubits, then CNOT(n-1, ancilla) and RY(w_l,n) on the ancilla.
    """
    B, n = x.shape
    L = weights.shape[0]
    nw = n + 1
    psi = zero_state(B, nw, x.device)
    for i in range(n):
        psi = rx(psi, x[:, i], i, nw)
        psi = rz(psi, x[:, i], i, nw)
    for i in range(n):
        psi = rx(psi, weights[0, i], i, nw)
        psi = rz(psi, weights[1, i], i, nw)
    for l in range(1, L):
        for i in range(n):
            psi = cnot(psi, i, (i + 1) % n, nw)
            psi = ry(psi, weights[l, i], i, nw)
            psi = rz(psi, weights[l, i], i, nw)
        psi = cnot(psi, n - 1, n, nw)
        psi = ry(psi, weights[l, n], n, nw)
    return expval_z(psi, n, nw)


def _entangle_ring(psi, n):
    if n == 2:
        return cnot(psi, 0, 1, n)
    if n > 2:
        for i in range(n):
            psi = cnot(psi, i, (i + 1) % n, n)
    return psi


def qrwkv_circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """QRWKV VQC (Fig. 4): AngleEmbedding(RX) + BasicEntanglerLayers(RX + CNOT ring).

    x [B, n], weights [D, n]; returns <Z> [B, n].
    """
    B, n = x.shape
    psi = zero_state(B, n, x.device)
    for i in range(n):
        psi = rx(psi, x[:, i], i, n)
    for l in range(weights.shape[0]):
        for i in range(n):
            psi = rx(psi, weights[l, i], i, n)
        psi = _entangle_ring(psi, n)
    return expval_z(psi, n, n)


def reupload_state(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Data re-uploading feature state |psi(x)> = S(x) W_D S(x) ... W_1 S(x) |0>.

    S(x) = prod_i RY(x_i) RZ(x_i);  W_l = prod_i Rot(w_l,i) followed by a CNOT ring.
    x [B, n], weights [D, n, 3]; returns the state [B, 2**n].
    A trainable block *after* the last encoding would cancel in the fidelity
    kernel, so the circuit always ends with an encoding layer.
    """
    B, n = x.shape
    psi = zero_state(B, n, x.device)

    def encode(p):
        for i in range(n):
            p = ry(p, x[:, i], i, n)
            p = rz(p, x[:, i], i, n)
        return p

    psi = encode(psi)
    for l in range(weights.shape[0]):
        for i in range(n):
            psi = rz(psi, weights[l, i, 0], i, n)
            psi = ry(psi, weights[l, i, 1], i, n)
            psi = rz(psi, weights[l, i, 2], i, n)
        psi = _entangle_ring(psi, n)
        psi = encode(psi)
    return psi


def reupload_expval(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return expval_z(reupload_state(x, weights), x.shape[1], x.shape[1])


def reupload_state_grouped(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Same circuit as `reupload_state`, for G independent parameter sets at once.

    x [G, N, n] angles, weights [G, D, n, 3]; returns states [G, N, 2**n].
    Consecutive single-qubit rotations on a wire are fused into one 2x2 matrix
    (RZ(x)RY(x) for the encoding, RZ RY RZ for the trainable Rot gate).
    """
    G, N, n = x.shape
    D = weights.shape[1]
    flat = x.reshape(G * N, n)
    psi = zero_state(G * N, n, x.device)
    # encoding matrices per wire: [G*N, 2, 2]
    enc = [_rz(flat[:, i]) @ _ry(flat[:, i]) for i in range(n)]
    for i in range(n):
        psi = _apply_1q(psi, enc[i], i, n)
    for l in range(D):
        for i in range(n):
            w = weights[:, l, i]  # [G, 3]
            rot = _rz(w[:, 2]) @ _ry(w[:, 1]) @ _rz(w[:, 0])  # [G, 2, 2]
            rot = rot.repeat_interleave(N, dim=0)
            psi = _apply_1q(psi, rot, i, n)
        psi = _entangle_ring(psi, n)
        for i in range(n):
            psi = _apply_1q(psi, enc[i], i, n)
    return psi.reshape(G, N, 2 ** n)


# ----------------------------------------------------------------------------
# nn.Module wrappers (quantum and matched classical surrogates)
# ----------------------------------------------------------------------------

class VQC(nn.Module):
    """R^n -> R^n map realised by a simulated circuit, or by a classical surrogate.

    The classical surrogate (`classical=True`) replaces the circuit with
    tanh(W x + b), W in R^{n x n}: a learned map with the same input/output
    width and a bounded output range ([-1, 1], like Pauli-Z expectations),
    used for the quantum-vs-classical ablation.
    """

    def __init__(self, n_qubits: int, depth: int, kind: str, classical: bool = False):
        super().__init__()
        self.n, self.depth, self.kind, self.classical = n_qubits, depth, kind, classical
        if classical:
            self.lin = nn.Linear(n_qubits, n_qubits)
        elif kind == "qasa":
            self.weights = nn.Parameter(torch.rand(depth, n_qubits + 1) * 2 * math.pi)
        elif kind == "qrwkv":
            self.weights = nn.Parameter(torch.rand(depth, n_qubits) * 2 * math.pi)
        elif kind == "reupload":
            self.weights = nn.Parameter(torch.rand(depth, n_qubits, 3) * 2 * math.pi)
        else:
            raise ValueError(kind)

    @property
    def hilbert_dim(self) -> int:
        if self.classical:
            return 0
        return 2 ** (self.n + 1) if self.kind == "qasa" else 2 ** self.n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.n)
        if self.classical:
            out = torch.tanh(self.lin(x))
        elif self.kind == "qasa":
            out = qasa_circuit(x.float(), self.weights)
        elif self.kind == "qrwkv":
            out = qrwkv_circuit(x.float(), self.weights)
        else:
            out = reupload_expval(x.float(), self.weights)
        return out.to(torch.float32).reshape(shape)
