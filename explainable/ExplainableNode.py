import torch
import numpy as np

class ExplainableNode:
    def __init__(self, should_compute_psm=False, should_compute_lcc=False):
        self.activation_values = {}
        self.gradient_sums = {}
        self.layer_names = {}
        self.hooks = []
        self.epoch_metrics = {}  # Change from list to dictionary
        self.should_compute_psm = should_compute_psm
        self.should_compute_lcc = should_compute_lcc

    def register_hooks(self, model, layers, layer_names):
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Register hooks and assign layer names
        for layer, name in zip(layers, layer_names):
            self.layer_names[layer] = name
            forward_hook = layer.register_forward_hook(self.forward_hook_fn)
            backward_hook = layer.register_backward_hook(self.backward_hook_fn)
            self.hooks.extend([forward_hook, backward_hook])

    def forward_hook_fn(self, module, input, output):
        self.activation_values[self.layer_names[module]] = output.detach()

    def backward_hook_fn(self, module, grad_input, grad_output):
        self.gradient_sums[self.layer_names[module]] = grad_output[0].detach().sum(dim=0)

    def calculate_metrics(self):
        # Check if layers are already in epoch_metrics; if not, initialize
        for layer_name in self.layer_names.values():
            if layer_name not in self.epoch_metrics:
                self.epoch_metrics[layer_name] = {'PSM': [], 'LCC': []}

        # Compute and append metrics
        if self.should_compute_psm:
            psm_metrics = self._compute_psm()
            for layer, value in psm_metrics.items():
                self.epoch_metrics[layer]['PSM'].append(value)

        if self.should_compute_lcc:
            lcc_metrics = self._compute_lcc()
            for layer, value in lcc_metrics.items():
                self.epoch_metrics[layer]['LCC'].append(value)

        return self.epoch_metrics

    def get_aggregated_epoch_metrics(self):
        # Aggregate metrics
        aggregated_metrics = {}
        for layer, metrics in self.epoch_metrics.items():
            aggregated_metrics[layer] = {metric: np.mean(values) for metric, values in metrics.items()}
        return aggregated_metrics

    def _compute_psm(self):
        return {name: torch.mean(torch.abs(grads)).item() for name, grads in self.gradient_sums.items()}

    def _compute_lcc(self):
        lcc_values = {}
        for name, activations in self.activation_values.items():
            activations_flat = activations.view(activations.size(0), -1)
            gradients_flat = self.gradient_sums[name].view(self.gradient_sums[name].size(0), -1)
            lcc = np.corrcoef(activations_flat.cpu().numpy(), gradients_flat.cpu().numpy(), rowvar=False)
            lcc_values[name] = lcc
        return lcc_values

    def clear_values(self):
        self.activation_values.clear()
        self.gradient_sums.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


