import json, base64, io, torch
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from torchvision import transforms
from .models_ai import model_asl, device, val_transforms

# Load model globally to avoid reloading on every request
asl_model = model_asl
asl_model.load_state_dict(torch.load("/home/faiz/Desktop/imageNet/backend/asl_cnn/finetune_modelv2.pth", map_location=torch.device(device)))
asl_model.eval()

SIGN_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def Home(request):
    return render(request, "base/home.html")


@csrf_exempt # Use proper CSRF in production
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        uri = data['image']
        header, encoded = uri.split(",", 1)
        data = base64.b64decode(encoded)
        
        # Preprocessing
        image = Image.open(io.BytesIO(data)).convert('L')
        transform = val_transforms
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = asl_model(input_tensor)
            idx = torch.argmax(output, 1).item()
            label = SIGN_CLASSES[idx]

        return JsonResponse({'label': label})