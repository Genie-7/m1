import torch
from model import GAT_MNIST, GATLayerMultiHead  # Ensure this import matches your actual model file name

# Load the saved model
saved_model = torch.load('best.pt', map_location=torch.device('cpu'))

# Create a new instance of the model
model = GAT_MNIST(num_features=3, num_classes=10)

# Load the saved state into the model
model.load_state_dict(saved_model)

# Now you can examine the model structure
for name, module in model.named_modules():
    if isinstance(module, GATLayerMultiHead):  # Adjust this if your GAT layer has a different name
        print(f"Layer {name} has {len(module.GAT_heads)} heads")