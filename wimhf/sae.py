"""Sparse autoencoder implementation for WIMHF."""

from __future__ import annotations

import copy
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SparseAutoencoder(nn.Module):
    """Batch Top-K sparse autoencoder with Matryoshka loss and signed activations."""

    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        *,
        aux_k: Optional[int] = None,
        multi_k: Optional[int] = None,
        dead_neuron_threshold_steps: int = 256,
        prefix_lengths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.m_total_neurons = m_total_neurons
        self.k_active_neurons = k_active_neurons

        self.aux_k = min(2 * k_active_neurons, m_total_neurons) if aux_k is None else aux_k
        self.multi_k = multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps

        self.prefix_lengths = prefix_lengths
        if self.prefix_lengths is not None:
            assert self.prefix_lengths[-1] == m_total_neurons, "Last prefix must equal total neurons"
            assert all(
                later > earlier for later, earlier in zip(self.prefix_lengths[1:], self.prefix_lengths[:-1])
            ), "Prefix lengths must be strictly increasing"

        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)

        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))

        self.steps_since_activation = torch.zeros(m_total_neurons, dtype=torch.long, device=DEVICE)
        self.register_buffer("threshold", torch.tensor(0.0, device=DEVICE))

        self.to(DEVICE)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_centered = x - self.input_bias
        pre_act = self.encoder(x_centered) + self.neuron_bias

        acts = pre_act
        batch_size = acts.shape[0]

        if self.training:
            scores = acts.abs().flatten()
            signed = acts.flatten()
            k_total = min(self.k_active_neurons * batch_size, scores.numel())
            _, idx = torch.topk(scores, k_total, dim=-1)
            signed_vals = signed[idx]
            activ_flat = torch.zeros_like(signed)
            activ_flat.scatter_(0, idx, signed_vals)
            activ = activ_flat.view_as(acts)
            self._update_threshold_(activ)
        else:
            mask = acts.abs() > self.threshold
            activ = torch.where(mask, acts, torch.zeros_like(acts))

        fired = (activ.abs().sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(-1)
        self.steps_since_activation += 1
        if fired.numel() > 0:
            self.steps_since_activation[fired] = 0

        # Multi-K reconstruction
        if self.multi_k is not None:
            _, multik_idx = torch.topk(pre_act.abs(), self.multi_k, dim=-1)
            multik_vals = pre_act.gather(-1, multik_idx)
            multik_activ = torch.zeros_like(pre_act).scatter_(-1, multik_idx, multik_vals)
            multik_recon = self.decoder(multik_activ) + self.input_bias
        else:
            multik_recon = None

        recon = self.decoder(activ) + self.input_bias

        aux_idx = aux_vals = None
        if self.aux_k is not None:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            dead_pre_act = pre_act * dead_mask
            aux_vals, aux_idx = torch.topk(dead_pre_act, self.aux_k, dim=-1)

        info = {
            "activations": activ,
            "multik_reconstruction": multik_recon,
            "aux_indices": aux_idx,
            "aux_values": aux_vals,
        }
        return recon, info

    @staticmethod
    def _normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        baseline = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
        return mse / baseline

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        info: Dict[str, torch.Tensor],
        aux_coef: float,
        multi_coef: float,
    ) -> torch.Tensor:
        activ = info["activations"]

        if self.prefix_lengths is None or len(self.prefix_lengths) == 1:
            main_l2 = self._normalized_mse(recon, x)
        else:
            terms = []
            decoder_w = self.decoder.weight
            for end in self.prefix_lengths:
                prefix_recon = activ[:, :end] @ decoder_w[:, :end].t() + self.input_bias
                terms.append(self._normalized_mse(prefix_recon, x))
            main_l2 = torch.stack(terms).mean()

        if multi_coef != 0 and info["multik_reconstruction"] is not None:
            main_l2 = main_l2 + multi_coef * self._normalized_mse(info["multik_reconstruction"], x)

        if self.aux_k is not None and info["aux_indices"] is not None:
            residual = x - recon.detach()
            aux_act = torch.zeros_like(activ).scatter_(-1, info["aux_indices"], info["aux_values"])
            aux_recon = self.decoder(aux_act)
            aux_loss = self._normalized_mse(aux_recon, residual)
            return main_l2 + aux_coef * aux_loss
        return main_l2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def adjust_decoder_gradient_(self) -> None:
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                proj = (self.decoder.weight * self.decoder.weight.grad).sum(dim=0, keepdim=True)
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor) -> None:
        self.input_bias.data = torch.median(data_sample, dim=0).values
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)

    @torch.no_grad()
    def _update_threshold_(self, activ: torch.Tensor, lr: float = 1e-2) -> None:
        if activ.numel() == 0:
            return
        pos = activ.abs() > 0
        if pos.any():
            min_positive = activ.abs()[pos].min()
            self.threshold.mul_(1 - lr).add_(lr * min_positive)

    def save(self, save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        config = {
            "input_dim": self.input_dim,
            "m_total_neurons": self.m_total_neurons,
            "k_active_neurons": self.k_active_neurons,
            "aux_k": self.aux_k,
            "multi_k": self.multi_k,
            "dead_neuron_threshold_steps": self.dead_neuron_threshold_steps,
            "prefix_lengths": self.prefix_lengths,
        }
        torch.save({"config": config, "state_dict": self.state_dict()}, save_path, pickle_module=pickle)
        print(f"[sae] saved model to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        min_epochs: int = 10,
        aux_coef: float = 1 / 32,
        multi_coef: float = 0.0,
        patience: int = 5,
        show_progress: bool = True,
        clip_grad: float = 1.0,
    ) -> Dict[str, List[float]]:
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(TensorDataset(X_val), batch_size=batch_size) if X_val is not None else None
        )

        self.initialize_weights_(X_train.to(DEVICE))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        history = {"train_loss": [], "val_loss": [], "dead_neuron_ratio": []}

        iterator = tqdm(range(n_epochs)) if show_progress else range(n_epochs)
        for epoch in iterator:
            self.train()
            train_losses = []

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(DEVICE)
                recon, info = self(batch_x)
                loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)

                optimizer.zero_grad()
                loss.backward()
                self.adjust_decoder_gradient_()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()
                self.normalize_decoder_()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history["train_loss"].append(avg_train_loss)

            dead_ratio = (
                (self.steps_since_activation > self.dead_neuron_threshold_steps).float().mean().item()
            )
            history["dead_neuron_ratio"].append(dead_ratio)

            avg_val_loss = float("nan")
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        batch_x = batch_x.to(DEVICE)
                        recon, info = self(batch_x)
                        val_loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < best_val_loss - 1e-6:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience and (epoch + 1) >= min_epochs:
                        if show_progress:
                            print(f"[sae] early stopping after {epoch + 1} epochs")
                        break

            if show_progress:
                iterator.set_postfix(
                    {
                        "train_loss": f"{avg_train_loss:.4f}",
                        "val_loss": f"{avg_val_loss:.4f}" if val_loader is not None else "N/A",
                        "dead_ratio": f"{dead_ratio:.3f}",
                        "threshold": f"{self.threshold.item():.2e}",
                    }
                )

        if val_loader is not None and best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            self.to(DEVICE)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = get_sae_checkpoint_name(self.m_total_neurons, self.k_active_neurons, self.prefix_lengths)
            self.save(os.path.join(save_dir, filename))

        return history

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    def get_activations(
        self,
        inputs,
        batch_size: int = 16_384,
        show_progress: bool = True,
    ) -> np.ndarray:
        self.eval()
        if isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float)
        elif isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be list, numpy array, or torch tensor")
        if inputs.dtype != torch.float:
            inputs = inputs.float()

        num_samples = inputs.shape[0]
        activations = []
        iterator = (
            tqdm(range(0, num_samples, batch_size), desc=f"Computing activations (batch={batch_size})")
            if show_progress
            else range(0, num_samples, batch_size)
        )

        with torch.no_grad():
            for start in iterator:
                batch = inputs[start : start + batch_size].to(DEVICE)
                _, info = self(batch)
                activations.append(info["activations"].cpu())

        return torch.cat(activations, dim=0).numpy()


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def get_sae_checkpoint_name(
    m_total_neurons: int,
    k_active_neurons: int,
    prefix_lengths: Optional[List[int]] = None,
) -> str:
    if prefix_lengths is None:
        return f"SAE_M={m_total_neurons}_K={k_active_neurons}.pt"
    prefix_str = "-".join(str(length) for length in prefix_lengths)
    return f"SAE_matryoshka_M={m_total_neurons}_K={k_active_neurons}_prefixes={prefix_str}.pt"


def load_model(path: str) -> SparseAutoencoder:
    checkpoint = torch.load(path, pickle_module=pickle, map_location=DEVICE)
    model = SparseAutoencoder(**checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"[sae] loaded model from {path}")
    return model
