import sys
import torch
import torch.nn as nn


def _print(s):
    print(s)
    sys.stdout.flush()


def get_latents(model, tokenizer, sequence, device):
    tokens = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embeds = outputs.hidden_states[-1].squeeze(0) # Get last hidden states
    return embeds



# General model freezing
def freeze_model(model: nn.Module):
    # Disable parameter updates for all layers
    for param in model.parameters():
        param.requires_grad = False



# For ProGen2 architecture
def apply_gptj_freezing(model, N_layers):
    def unfreeze_n_layers(model, N_layers):
        # Count number of encoder layers
        model_layers = len(model.transformer.h)
        for i, h in enumerate(model.transformer.h):
            if i >= model_layers - N_layers:
                for module in h.attn.modules():
                    for param in module.parameters():
                        param.requires_grad = True

    def check_frozen_model(model, N_layers: int):
        """
        Verify that only the last N_layers of model.transformer.h are unfrozen.
        Source: https://github.com/enijkamp/progen2/blob/main/progen/modeling_progen.py
        """
        model_layers = len(model.transformer.h)
        frozen_layers = 0
        unfrozen_layers = 0
        for i, h in enumerate(model.transformer.h):
            if i >= model_layers - N_layers:  # should be unfrozen
                if any(param.requires_grad for param in h.parameters()):
                    unfrozen_layers += 1
                else:
                    print(f"Layer {i} has all parameters frozen, but it should be unfrozen.")
            else:  # should be frozen
                if any(param.requires_grad for param in h.parameters()):
                    print(f"Layer {i} is not frozen, but it should be frozen.")
                else:
                    frozen_layers += 1

        assert frozen_layers == model_layers - N_layers and unfrozen_layers == N_layers, \
            f"frozen layers: {frozen_layers}, unfrozen layers: {unfrozen_layers}"

        print(f"frozen layers: {frozen_layers}, unfrozen layers: {unfrozen_layers}")

    freeze_model(model)
    unfreeze_n_layers(model, N_layers)
    check_frozen_model(model, N_layers)





# For RDM-based architectures
def apply_rdm_freezing(model: nn.Module, N_layers: int, model_type: str):
    """
    Freeze all layers except last N for esm-like architectures

    Args:
        model (nn.Module): model to freeze
        N_layers (int): num encoder layers to unfreeze
        model_type (str): one of {"esm", "evoflow", "dplm"}
    """

    # choose encoder layers based on the model type
    if model_type == "dplm":
        encoder_layers = model.net.esm.encoder.layer
    elif model_type in ("esm", "evoflow"):
        encoder_layers = model.esm.encoder.layer
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    def unfreeze_n_layers(layers, N_layers: int):
        model_layers = len(layers)
        for i, layer in enumerate(layers):
            if i >= model_layers - N_layers:
                for module in layer.attention.self.key.modules():
                    for param in module.parameters():
                        param.requires_grad = True
                for module in layer.attention.self.query.modules():
                    for param in module.parameters():
                        param.requires_grad = True
                for module in layer.attention.self.value.modules():
                    for param in module.parameters():
                        param.requires_grad = True

    def check_model(layers, N_layers: int):
        model_layers = len(layers)
        frozen_layers = 0
        unfrozen_layers = 0

        for i, layer in enumerate(layers):
            if i >= model_layers - N_layers:
                layer_frozen = True
                for module in layer.attention.self.key.modules():
                    if any(param.requires_grad for param in module.parameters()):
                        layer_frozen = False
                for module in layer.attention.self.query.modules():
                    if any(param.requires_grad for param in module.parameters()):
                        layer_frozen = False
                for module in layer.attention.self.value.modules():
                    if any(param.requires_grad for param in module.parameters()):
                        layer_frozen = False
                
                if layer_frozen:
                    print(f"layer {i} has all parameters frozen, but it should be unfrozen.")
                else:
                    unfrozen_layers += 1
            else:
                if any(param.requires_grad for param in layer.parameters()):
                    print(f"layer {i} is not frozen, but it should")
                else:
                    frozen_layers += 1

        assert (frozen_layers == model_layers - N_layers) and (unfrozen_layers == N_layers), \
            f"frozen layers: {frozen_layers}, unfrozen layers: {unfrozen_layers}"


    freeze_model(model)
    unfreeze_n_layers(encoder_layers, N_layers)
    check_model(encoder_layers, N_layers)
