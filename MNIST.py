import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pygame 

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root='data',       # where to save the data
    train=True,        # training set
    download=True,     # download if not already present
    transform=transform  # apply transformation
)

# Download the MNIST test dataset
test_data = datasets.MNIST(
    root='data',
    train=False,       # test set
    download=True,
    transform=transform
)
# If the output when running this block is 100%, the dataset was downloaded and stored correctly

# Get one sample and print its label, then show the image using matplotlib
#image_tensor, label = train_data[0]  # get the first sample
#print(label)  # print the label

# Visualize the first image in the training set
# image_tensor = train_data[0][0]
# image_to_plot = image_tensor.squeeze()
# plt.imshow(image_to_plot, cmap='gray')
# plt.axis('off')
# plt.show()
#
# Define a simple neural network for digit classification
class DigitClassifier(nn.Module):  # 1. Inherit from nn.Module
    def __init__(self):
        super().__init__()  # 2. Init the base class
        self.hidden = nn.Linear(784, 128)  # Hidden layer: 784 input features (28x28 pixels) to 128
        self.output = nn.Linear(128, 10)   # Output layer: 128 features to 10 classes (digits 0-9)
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x  

# Create a DataLoader for batching the training data
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# Instantiate the model
instance = DigitClassifier()
# Define the optimizer
optimizer = torch.optim.SGD(instance.parameters(), lr=0.015)

# Training loop (one epoch)
for images, labels in train_dataloader:
    # Forward pass: compute model output
    model = instance(images)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model, labels)
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss)

# Create a DataLoader for batching the test data



test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
correct = 0
total = 0
# Disable gradient calculation for evaluation
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = instance(images)  # Get model predictions
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Print and visualize incorrect predictions
        # for i in range(len(labels)):
        #     if predicted[i] != labels[i]:
        #         print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
        #         image_tensor = images[i][0]
        #         image_to_plot = image_tensor.squeeze()
        #         plt.imshow(image_to_plot, cmap='gray')
        #         plt.axis('off')
        #         plt.show()
# Calculate and print accuracy
accuracy = 100 * correct / total
#print(accuracy)


# pygame setup
import numpy as np
import cv2

pygame.init()
screen = pygame.display.set_mode((280, 320))
pygame.display.set_caption("Draw a digit")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 32)
button_font = pygame.font.SysFont("Arial", 24)
reset_button = pygame.Rect(10, 285, 80, 25)

# Create a surface for drawing
drawing_surface = pygame.Surface((280, 280))
drawing_surface.fill((0, 0, 0))  # Black background

running = True
predicted_digit = None
while running:
    screen.fill((0, 0, 0))  # Clear screen
    screen.blit(drawing_surface, (0, 0))  # Blit drawing surface

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                drawing_surface.fill((0, 0, 0))  # Clear drawing surface
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if reset_button.collidepoint(event.pos):
                drawing_surface.fill((0, 0, 0))  # Clear drawing surface

    if pygame.mouse.get_pressed()[0]:
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[1] < 280:
            pygame.draw.circle(drawing_surface, (255, 255, 255), mouse_pos, 4)

    # Process the drawing
    drawing_array = pygame.surfarray.array3d(drawing_surface)
    gray_image = cv2.cvtColor(np.transpose(drawing_array, (1, 0, 2)), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        padded_digit = np.pad(resized_digit, ((4, 4), (4, 4)), "constant", constant_values=0)
        normalized_digit = padded_digit / 255.0
        input_tensor = torch.tensor(normalized_digit, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            input_tensor_flat = input_tensor.view(1, -1)
            output = instance(input_tensor_flat)
            _, predicted = torch.max(output.data, 1)
            predicted_digit = predicted.item()

    # Draw prediction text if it exists
    if predicted_digit is not None:
        pygame.draw.rect(screen, (0, 0, 0), (0, 280, 280, 40))
        text = font.render(str(predicted_digit), True, (255, 0, 0))
        screen.blit(text, (200, 285))
    # make a box around the drawing aread
    pygame.draw.rect(screen, (255, 0, 0), (40, 40, 280-80, 280-80), 2) 
    # Draw reset button
    pygame.draw.rect(screen, (200, 200, 200), reset_button)
    button_text = button_font.render("Reset", True, (0, 0, 0))
    screen.blit(button_text, (reset_button.x + 10, reset_button.y + 2))


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
