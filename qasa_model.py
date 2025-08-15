import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

n_qubits = 8
n_layers = 4

dev = qml.device("default.qubit", wires=n_qubits + 1)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # inputs: shape [n_qubits]
    # print("QNODE INPUT SHAPE:", inputs.shape)
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    for l in range(1, n_layers):
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        qml.CNOT(wires=[n_qubits - 1, n_qubits])
        qml.RY(weights[l, -1], wires=n_qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_shape = (n_layers, n_qubits + 1)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {"weights": self.weight_shape})
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.norm = nn.LayerNorm(n_qubits)
        self.output_proj = nn.Linear(n_qubits, output_dim)

    def forward(self, x, timestep):
        # x: [B, input_dim], timestep: scalar or [B]
        x_proj = self.norm(torch.tanh(self.input_proj(x)))

        if timestep is not None:
            if timestep.dim() == 0:
                timestep_exp = timestep.expand_as(x_proj)
            else:
                timestep_exp = timestep.view(-1, 1).expand_as(x_proj)
            x_proj = x_proj + timestep_exp

        # Process each sample in the batch individually
        batch_size = x_proj.shape[0]
        q_out_list = [self.qlayer(x_proj[i]) for i in range(batch_size)]
        q_out = torch.stack(q_out_list)

        out = self.output_proj(q_out)  # [B, output_dim]

        if self.input_dim == self.output_dim:
            return x + out
        else:
            return out

class QASARegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1
        )
        self.q_layer = QuantumLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, timestep=None):
        # x: [B, S] or [B, S, 1]
        if x.dim() == 3:
            x = x.squeeze(-1)  # [B, S]
        # x should now be [B, S] - we need to apply linear layer along the last dimension
        # But input_proj expects [B, S, input_dim], so we need to unsqueeze
        x = x.unsqueeze(-1)  # [B, S, 1]
        x = self.input_proj(x)  # [B, S, D]
        x = self.encoder(x)     # [B, S, D]
        x = x.mean(dim=1)       # [B, D]
        x = self.q_layer(x, timestep if timestep is not None else torch.tensor(0.0, device=x.device))
        return self.fc(x)       # [B, 1]

class QASAWrapper(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, seq_len=96, output_dim=1):
        super().__init__()
        self.model = QASARegressor(input_dim, hidden_dim, seq_len, output_dim)

    def forward(self, x, timestep=None):
        return self.model(x, timestep).squeeze(-1)  # [B]

    def compute_custom_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets, reduction='none') 