import torch
import torch.tensor as Tensor

def mask_negative_log_likelihood_loss(\
        input_vector,
        target_vector,
        mask_vector, 
        used_device
    ):
    if type(input_vector) is not Tensor:
        raise TypeError(\
                "Expected input_vector as Tensor but got as {0}"
                .format(type(input_vector))
            )
    if type(target_vector) is not Tensor:
        raise TypeError(\
                "Expected target_vector as Tensor but got as {0}"
                .format(type(target_vector))
            )
    if type(mask_vector) is not Tensor:
        raise TypeError(\
                "Expected mask_vector as Tensor but got as {0}"
                .format(type(mask_vector))
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
    
    n_total = mask_vector.sum()
    cross_entropy = -torch.log(\
            torch.gather(input_vector, 1, target_vector.view(-1, 1)).squeeze(1)
        )
    meaned_loss = cross_entropy.masked_select(mask_vector).mean()
    meaned_loss = meaned_loss.to(used_device)
    return meaned_loss, n_total.item()