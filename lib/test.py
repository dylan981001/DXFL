import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

# Define the original loss function
def custom_contrastive_loss_original(features, class_groups, device, tau=0.1):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)
    
    batch_size, _ = features.size()
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)

    features_label = torch.tensor(features_label, device=device)

    for i in range(len(features)):
        pos_numerators = []
        pos_denominator = torch.tensor(0.0, device=device)

        for j in range(len(features)):
            if i != j and features_label[i] == features_label[j]:
                pos_sim = similarity_matrix[i][j] / tau
                pos_numerators.append(torch.exp(pos_sim))
                pos_denominator += torch.exp(pos_sim)

        if len(pos_numerators) != 0:
            for k in range(len(pos_numerators)):
                print(f'original: {pos_numerators}')
                loss -= torch.log((pos_numerators[k]) / (pos_denominator + 1e-9))

    return loss

# Define the optimized loss function
def custom_contrastive_loss_optimized(features, class_groups, device, tau=0.1):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)

    batch_size, _ = features.size()
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)
    
    features_label = torch.tensor(features_label, device=device)
    
    labels_equal = features_label.unsqueeze(0) == features_label.unsqueeze(1)
    labels_not_equal = ~labels_equal
    
    pos_mask = labels_equal & (torch.eye(batch_size, device=device) == 0)
    pos_sim = similarity_matrix[pos_mask] / tau
    pos_numerators = torch.exp(pos_sim)
    pos_denominators = pos_numerators.sum() + 1e-9

    if pos_numerators.numel() > 0:
        print(f'optimized: {pos_numerators}')
        loss -= torch.sum(torch.log(pos_numerators / pos_denominators))

    return loss / batch_size

# Example usage with CIFAR-100 data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(cifar100, batch_size=32, shuffle=True)

# Get a batch of data
data_iter = iter(dataloader)
images, labels = data_iter.next()

# Example class_groups
class_groups = {}
for idx, label in enumerate(labels):
    if label.item() not in class_groups:
        class_groups[label.item()] = []
    class_groups[label.item()].append(idx)

# Flatten the images to create feature vectors
features = images.view(images.size(0), -1).to(device)

# Measure the execution time for the original loss function
start_time = time.time()
loss_original = custom_contrastive_loss_original(features, class_groups, device)
end_time = time.time()
print("Original Loss:", loss_original.item())
print("Original Execution Time:", end_time - start_time)

# Measure the execution time for the optimized loss function
start_time = time.time()
loss_optimized = custom_contrastive_loss_optimized(features, class_groups, device)
end_time = time.time()
print("Optimized Loss:", loss_optimized.item())
print("Optimized Execution Time:", end_time - start_time)
