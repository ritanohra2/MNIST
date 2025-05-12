# MNIST Test Project

This project demonstrates loading, visualizing, training, and evaluating a simple neural network on the MNIST dataset using PyTorch and torchvision.

## Project Structure

- `notebook_mnist.ipynb`: Jupyter notebook with code for loading, visualizing, batching, training, and testing MNIST data.
- `data/`: Directory containing the MNIST dataset (downloaded automatically).
- `requirements.txt`: List of required Python packages.


## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open `notebook_mnist.ipynb` in Jupyter or VS Code and run the cells to:
   - Import libraries
   - Download and load the MNIST dataset
   - Visualize sample images
   - Create a DataLoader for batching
   - Define and train a simple neural network classifier
   - Evaluate the model and visualize incorrect predictions

## Notes

- The dataset will be downloaded automatically to the `data/` directory if not already present.
- Incorrect predictions are displayed with their predicted and actual labels, along with the image.

## License

MIT License