# Assuming your test data is in 'finalData/test' and uses the same normalization as training and validation

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), test_transforms)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)


# Ensure you have the test DataLoader 'test_loader' set up
# Set model to evaluation mode
model.eval()

# Track the number of correct predictions and total predictions
correct = 0
total = 0

# No gradient is needed for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate the percentage accuracy
accuracy = (correct / total) * 100
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
