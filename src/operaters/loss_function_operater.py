import torch

def mask_negative_log_likelihood_loss(\
        input_tensor,
        target_tensor,
        mask_tensor, 
        used_device
    ):
    if type(input_tensor) is not torch.Tensor:
        raise TypeError(\
                "Expected input_tensor as torch.Tensor but got as {0}"
                .format(type(input_tensor))
            )
    if type(target_tensor) is not torch.Tensor:
        raise TypeError(\
                "Expected target_tensor as torch.Tensor but got as {0}"
                .format(type(target_tensor))
            )
    if type(mask_tensor) is not torch.Tensor:
        raise TypeError(\
                "Expected mask_tensor as torch.Tensor but got as {0}"
                .format(type(mask_tensor))
            )
    if type(used_device) is not str:
        raise TypeError(\
                "Expected used_device as str but got as {0}"
                .format(type(used_device))
            )
    used_devices = ["cpu", "gpu"]
    if used_device not in used_devices:
        error_message = "Unknown device. Only accepted "
        for device in used_devices:
            error_message = error_message + device + " "
        raise ValueError(error_message)
    
    n_total = mask_tensor.sum()
    cross_entropy = -torch.log(\
            torch.gather(input_tensor, 1, target_tensor.view(-1, 1)).squeeze(1)
        )
    meaned_loss = cross_entropy.masked_select(mask_tensor).mean()
    meaned_loss = meaned_loss.to(used_device)
    return meaned_loss, n_total.item()