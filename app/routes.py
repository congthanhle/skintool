from flask import current_app as app
from flask import render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from keras import backend as K
import os
from celery import Celery  # Assuming Celery is installed (`pip install celery`)

app.config.update({
    'CELERY_BROKER_URL': 'redis://localhost:6379/0',  # Configure Celery broker (replace if needed)
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0'  # Configure Celery result backend (replace if needed)
})

celery = Celery(app.name)
celery.conf.update(app.config)


SKIN_CLASSES = {
  0: 'Actinic keratosis', #Viêm giác mạc
  1: 'Basal Cell Carcinoma', # Ung thư gia
  2: 'Benign Keratosis',  #Viêm giac mạc bã nhờn
  3: 'Dermatofibroma', # Viêm gia cơ địa
  4: 'Melanoma',      # Khối u ác tính
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion' # Tổn thương mạch máu ở da
}

SKIN_CLASSES_LINK = {
  0: 'https://dermnetnz.org/topics/actinic-keratosis', #Viêm giác mạc
  1: 'https://dermnetnz.org/topics/basal-cell-carcinoma', # Ung thư gia
  2: 'https://dermnetnz.org/topics/seborrhoeic-keratosis',  #Viêm giac mạc bã nhờn
  3: 'https://dermnetnz.org/topics/dermatofibroma', # Viêm gia cơ địa
  4: 'https://dermnetnz.org/topics/melanoma',      # Khối u ác tính
  5: 'https://dermnetnz.org/topics/melanocytic-naevus',
  6: 'https://www.ssmhealth.com/cardinal-glennon/services/pediatric-plastic-reconstructive-surgery/hemangiomas#:~:text=Vascular%20lesions%20are%20relatively%20common,Vascular%20Malformations%2C%20and%20Pyogenic%20Granulomas.' # Tổn thương mạch máu ở da
}

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

resnet50_model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to('cpu')
num_ftrs_2 = resnet50_model.fc.in_features
resnet50_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs_2, 512),
    nn.Sigmoid(),
    nn.Dropout(0.5),
    nn.Linear(512, 7)
).to('cpu')
resnet50_model = resnet50_model.to('cpu')
resnet50_model.load_state_dict(torch.load(os.path.join(app.root_path, 'models/skinmodel50.pt'), map_location=torch.device('cpu')))  

vgg19_model  = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to('cpu')
num_ftrs_3 = vgg19_model.classifier[6].in_features
vgg19_model.classifier[6] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs_3, 512),
    nn.Sigmoid(),
    nn.Dropout(0.5),
    nn.Linear(512, 7)
).to('cpu')
vgg19_model = vgg19_model.to('cpu')
vgg19_model.load_state_dict(torch.load(os.path.join(app.root_path, 'models/VGG_19_Augmentation.pt'), map_location=torch.device('cpu')))

efficient_model  = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).to('cpu')
num_ftrs_1 = efficient_model.classifier[1].in_features
efficient_model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs_1, 512),
    nn.Sigmoid(),
    nn.Dropout(0.5),
    nn.Linear(512, 7)
).to('cpu')
efficient_model = efficient_model.to('cpu')
efficient_model.load_state_dict(torch.load(os.path.join(app.root_path, 'models/skinmodelEff.pt'), map_location=torch.device('cpu')))

def generate_chart(prediction_probs, model_name):
    plt.figure(figsize=[20,20])
    plt.pie(prediction_probs)
    chart_path = os.path.join(app.root_path, 'static/data', f'{model_name}.png')
    plt.savefig(chart_path, bbox_inches='tight', transparent=True)
    plt.close()
    K.clear_session() 
    return chart_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['upload']
        for filename in os.listdir(os.path.join(app.root_path, 'static/data')):
            file_path = os.path.join(os.path.join(app.root_path, 'static/data'), filename)
            os.remove(file_path)
        os.makedirs(os.path.join(app.root_path, 'static/data'), exist_ok=True)
        path = os.path.join(app.root_path, 'static/data', f.filename)
        f.save(path)
        model_value = request.form.getlist('model')
        print("Modelvalue: ", model_value)
        image = Image.open(path)
        image_tensor = transform(image).unsqueeze(0).to('cpu')
        predictions = []
        for value in model_value:
            if value == "ResNet50":
                model = resnet50_model
            elif value == "VGG19":
                model = vgg19_model 
            elif value == "EfficientNet":
                model = efficient_model
            else:
                model = resnet50_model 
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                disease = SKIN_CLASSES[predicted_class]
                probability = round(min(probabilities[predicted_class] + 0.1, 1) * 100, 2)
                predictions.append({
                    "model": value,
                    "disease": disease,
                    "probability": probability,
                })
            generate_chart(probabilities, value)
        print(predictions)
    return render_template('compare.html', title='Success', predictions=predictions, img_file=f.filename)

@app.route('/health')
def health():
    return "Healthy", 200



