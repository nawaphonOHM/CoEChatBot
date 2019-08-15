from src.operaters.loss_function_operater import mask_negative_log_likelihood_loss
import torch
import random

input_tensor = torch.randint(100, (3, 3, 3))
print("An input tensor is => ", end="")
print(input_tensor)

target_tensor = torch.randint(100, (3, 3, 3))
print("A target tensor is => ", end="")
print(target_tensor)

mask_tensor = torch.LongTensor(\
    [[[(0 if random.random() < 0.5 else 1) for _ in range(3)] for _ in range(3)] for _ in range(3)])
print("A mask tensor is => ", end="")
print(mask_tensor)

print(torch.gather(input_tensor[0], 1, target_tensor[0].view(-1, 1)))

# loss, item = mask_negative_log_likelihood_loss(\
#     input_tensor[0], target_tensor[0], mask_tensor[0], "cpu")

