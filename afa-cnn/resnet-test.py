from keras.applications.resnet50 import ResNet50

# Load the ResNet50 model
model = ResNet50(weights='imagenet')

# Print details of each layer
for i, layer in enumerate(model.layers):
    print(f"Layer {i}:")
    print(f"    Name: {layer.name}")
    print(f"    Type: {layer.__class__.__name__}")
    print(f"    Input Shape: {layer.input_shape}")
    print(f"    Output Shape: {layer.output_shape}")
    print("\n")
