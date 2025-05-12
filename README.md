# MNIST Test Project

This project demonstrates loading, visualizing, and batching the MNIST dataset using PyTorch and torchvision.

## Project Structure

- `notebook_mnist.ipynb`: Jupyter notebook with code for loading, visualizing, and batching MNIST data.
- `data/`: Directory containing the MNIST dataset (downloaded automatically).
- `requirements.txt`: List of required Python packages.

class DigitClassifier(nn.Module):
[Image: 1×28×28]
   ↓ flatten
[Vector: 784]
   ↓ Linear(784 → 128)
   ↓ ReLU
[Vector: 128]
   ↓ Linear(128 → 10)
   ↓ (raw scores)
[Vector: 10]


## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Open `notebook_mnist.ipynb` in Jupyter or VS Code and run the cells to:
   - Import libraries
   - Download and load the MNIST dataset
   - Visualize sample images
   - Create a DataLoader for batching

## Notes

- The dataset will be downloaded automatically to the `data/` directory if not already present.

## License

TBD