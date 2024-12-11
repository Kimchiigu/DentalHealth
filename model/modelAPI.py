from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from fastapi.middleware.cors import CORSMiddleware

# Create a FastAPI app (declare only once)
app = FastAPI()

# Define CORS origins
origins = [
    "http://localhost:3000",  # Frontend running on localhost:3000
    "http://localhost:8000",  # Backend on localhost:8000 (not usually needed)
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the custom model class with exact structure
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to load the model
def load_model():
    num_classes = 6  # Update with the correct number of classes
    model = CustomModel(num_classes)
    try:
        model.load_state_dict(torch.load("model_complete.pth", map_location=torch.device('cpu')), strict=False)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
    return model

# Initialize the model globally
model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 expects 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['Calculus', 'Caries', 'Gingivitis', 'Ulcers', 'Tooth Discoloration', 'Hypodontia']  # Update if needed

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image
        image = Image.open(file.file).convert("RGB")
        
        # Apply image transformations
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        
        # Return the predicted class
        return {"predicted_class": predicted_class}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # The module reference should match the file name if not run directly as a script
    uvicorn.run("modelAPI:app", host="0.0.0.0", port=8000, reload=True)
