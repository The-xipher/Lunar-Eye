import torch
import torchvision
from torchvision import transforms, models
from torch import nn
import cv2
import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
# Constants 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Model Definition
class Model1(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model1, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)  # Reshape to maintain batch and sequence dimensions
        x_lstm, _ = self.lstm(x, None)
        x_lstm = x_lstm[:, -1, :]  # Only consider the last output of the LSTM
        x_dp = self.dp(x_lstm)
        logits = self.linear1(x_dp)
        return fmap, logits
    
# Load Model
def load_model(model_path):
    if torch.cuda.is_available():
        map_location = 'cuda'
    else:
        map_location = 'cpu'
    
    model = Model1(2)
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    if torch.cuda.is_available():
        model.cuda()  # Move model to GPU if available
    return model

# Preprocess Image

# Preprocess Image with CLAHE enhancement
def preprocess_frame(frame):
    # Convert frame to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE on the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    
    # Convert back to BGR color space
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply other transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    frame_tensor = transform(enhanced_frame).unsqueeze(0).unsqueeze(0)  # Add batch_size and seq_length dimensions
    return frame_tensor



def predict(model, frame_path):
    frame = cv2.imread(frame_path)
    # Enhance image
    enhanced_frame = preprocess_frame(frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhanced_frame_tensor = enhanced_frame.to(device)
    with torch.no_grad():
        fmap, logits = model(enhanced_frame_tensor)
        probabilities = nn.Softmax(dim=1)(logits)
        _, prediction = torch.max(probabilities, 1)
        confidence = probabilities[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence


# Main function
def main():
    # Paths
    model_path = "model\model_97_acc_80_frames_FF_data.pt"
    input_root = "upload"  # Change input directory name
    out_root = "temp_frame"  # Change output directory name
    num_frames = 80  # Number of frames to extract

    # Load model
    model = load_model(model_path)

    # Iterate through input directory
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                out_path = video_path.replace(input_root, out_root)[:-4] + '/'  # Output directory path
                os.makedirs(out_path, exist_ok=True)  # Create output directory if not exists

                # Read video and extract frames
                vidcap = cv2.VideoCapture(video_path)
                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int)
                frameno=0
                sumval=0
                frames=[]
                confidencelist=[]
                success, image = vidcap.read()
                count = 0
                while success:
                    if count not in frame_idxs:
                        success, image = vidcap.read()
                        count += 1
                        continue

                    # Save frame as JPEG file
                    frame_path = os.path.join(out_path, f'frame{count}.jpg')
                    cv2.imwrite(frame_path, image)

                    # Perform prediction on the frame
                    prediction, confidence = predict(model, frame_path)
                    sumval=sumval+confidence
                    frameno += 1

                    success, image = vidcap.read()
                    count += 1
                    frames.append(frameno)
                    confidencelist.append(confidence)
                vidcap.release()
                accuracy=round((sumval/frameno),2)
                print("* Total no of frames: ",frameno)
                print("* Estimated Accuracy: ",accuracy/100)
                if accuracy>=60:
                    print("* Estimated Result: Video Might be Real")
                else:
                    print("* Estimated Result: Video Might be Fake")
    plt.plot(frames, confidencelist, color='red')
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Confidence of Each Frame')

    # Save the graph image to the 'graph' folder
    graph_folder = './graph'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
    graph_path = os.path.join(graph_folder, 'confidence_graph.png')
    plt.savefig(graph_path)
if __name__ == "__main__":
    main()
