"""Check the PyTorch state-vector simulator against PennyLane (default.qubit)."""

import math
import os
import sys

import numpy as np
import pennylane as qml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qnlg import quantum as Q  # noqa: E402

torch.manual_seed(0)
TOL = 1e-5


def pl_qasa(x, w):
    n = len(x)
    L = w.shape[0]
    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev)
    def circ():
        for i in range(n):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(n):
            qml.RX(w[0, i], wires=i)
            qml.RZ(w[1, i], wires=i)
        for l in range(1, L):
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])
                qml.RY(w[l, i], wires=i)
                qml.RZ(w[l, i], wires=i)
            qml.CNOT(wires=[n - 1, n])
            qml.RY(w[l, -1], wires=n)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return np.array(circ())


def pl_qrwkv(x, w):
    n = len(x)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ():
        qml.AngleEmbedding(x, wires=range(n), rotation="X")
        qml.BasicEntanglerLayers(w, wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return np.array(circ())


def pl_reupload_state(x, w):
    n = len(x)
    dev = qml.device("default.qubit", wires=n)

    def ring():
        if n == 2:
            qml.CNOT(wires=[0, 1])
        elif n > 2:
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])

    def enc():
        for i in range(n):
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)

    @qml.qnode(dev)
    def circ():
        enc()
        for l in range(w.shape[0]):
            for i in range(n):
                qml.Rot(w[l, i, 0], w[l, i, 1], w[l, i, 2], wires=i)
            ring()
            enc()
        return qml.state()

    return np.array(circ())


def test_qasa():
    for n in (2, 4, 8):
        x = torch.rand(3, n) * 2 - 1
        w = torch.rand(4, n + 1) * 2 * math.pi
        ours = Q.qasa_circuit(x, w).numpy()
        for b in range(3):
            ref = pl_qasa(x[b].numpy(), w.numpy())
            assert np.allclose(ours[b], ref, atol=TOL), (n, ours[b], ref)


def test_qrwkv():
    for n in (2, 4, 6):
        x = torch.rand(3, n) * 2 - 1
        w = torch.rand(2, n) * 2 * math.pi
        ours = Q.qrwkv_circuit(x, w).numpy()
        for b in range(3):
            ref = pl_qrwkv(x[b].numpy(), w.numpy())
            assert np.allclose(ours[b], ref, atol=TOL), (n, ours[b], ref)


def test_reupload_state_and_kernel():
    for n in (2, 4):
        x = torch.rand(3, n) * 2 - 1
        w = torch.rand(2, n, 3) * 2 * math.pi
        ours = Q.reupload_state(x, w).numpy()
        refs = [pl_reupload_state(x[b].numpy(), w.numpy()) for b in range(3)]
        for b in range(3):
            # identical up to numerical precision (no global-phase freedom: same gates)
            assert np.allclose(ours[b], refs[b], atol=TOL)
        # fidelity kernel is symmetric, PSD, unit diagonal
        K = np.abs(ours.conj() @ ours.T) ** 2
        assert np.allclose(np.diag(K), 1.0, atol=TOL)
        assert np.allclose(K, K.T, atol=TOL)
        assert np.linalg.eigvalsh(K).min() > -1e-6


def test_grouped_matches_single():
    G, N, n = 3, 5, 4
    x = torch.rand(G, N, n) * 2 - 1
    w = torch.rand(G, 2, n, 3) * 2 * math.pi
    grouped = Q.reupload_state_grouped(x, w)
    for g in range(G):
        single = Q.reupload_state(x[g], w[g])
        assert torch.allclose(grouped[g], single, atol=TOL)


def test_gradients_flow():
    x = torch.rand(5, 4, requires_grad=True)
    vqc = Q.VQC(4, 2, "qrwkv")
    vqc(x).sum().backward()
    assert x.grad is not None and vqc.weights.grad is not None
    assert torch.isfinite(vqc.weights.grad).all()


if __name__ == "__main__":
    test_qasa()
    test_qrwkv()
    test_reupload_state_and_kernel()
    test_grouped_matches_single()
    test_gradients_flow()
    print("all quantum simulator tests passed")
