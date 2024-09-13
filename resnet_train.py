import argparse
import os
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
from datasets import load_dataset
from resnet_attack_todo import ResnetPGDAttacker

RESULTS_DIR = 'results'
RESULTS_PATH = os.path.join(RESULTS_DIR, 'test_3')
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

SEED = 1234
torch.manual_seed(SEED)

EPS = 8/255
ALPHA = 2/255
STEPS = 20
BATCH_SIZE = 64
EPOCHS = 5

print('Loading model...')
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()

print('Loading data...')
# Load original dataset
ds = load_dataset("ILSVRC/imagenet-1k", split="train",
                  streaming=True, trust_remote_code=True)


def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example


# Filter out grayscale images
ds = ds.filter(lambda example: example['image'].mode == 'RGB')
ds = ds.map(preprocess_img)
ds = ds.shuffle(seed=SEED)
ds = ds.take(BATCH_SIZE * EPOCHS)

# Load adversarial images and labels from the result file
attack_results = torch.load(RESULTS_PATH)
adv_images = attack_results['adv_images']
adv_labels = attack_results['labels']

# Check adversarial data
print(f"Adversarial images shape: {adv_images.shape}")
print(f"Adversarial labels shape: {adv_labels.shape}")

# Prepare original and adversarial datasets
original_images = []
original_labels = []

for batch in DataLoader(ds, batch_size=BATCH_SIZE):
    images = batch['image']
    labels = batch['label']
    original_images.append(images)
    original_labels.append(labels)

# Combine original and adversarial data
original_images = torch.cat(original_images)
original_labels = torch.cat(original_labels)

# Verify shapes
print(f"Original images shape: {original_images.shape}")
print(f"Original labels shape: {original_labels.shape}")

combined_images = torch.cat([original_images, adv_images]).float()
combined_labels = torch.cat([original_labels, adv_labels]).long()

# Check shapes and types
print(f"Combined images shape: {combined_images.shape}")
print(f"Combined labels shape: {combined_labels.shape}")

combined_dataset = TensorDataset(combined_images, combined_labels)
combined_loader = DataLoader(
    combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define training parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

print('Starting adversarial training...')
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(combined_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(combined_loader)}')

print('Adversarial training complete.')
# Save the fine-tuned model
torch.save(model.state_dict(), os.path.join(
    RESULTS_DIR, 'fine_tuned_resnet.pth'))

