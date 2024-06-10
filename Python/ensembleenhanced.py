import torch
import torchvision
from torchvision import transforms, models
from torch import nn
import cv2
import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model 
import os
from tqdm import tqdm
# Constants
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}
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
class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

# Create a MesoNet class using the Classifier
class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)
    
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
    num_frames = 200  # Number of frames to extract

    # Load model
    model = load_model(model_path)

    framesresnext = []
    confidencelistresnext = []
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
                    print(f"Frame {frameno}: Prediction - {prediction}, Confidence - {confidence:.2f}%")
                    frameno += 1

                    success, image = vidcap.read()
                    count += 1
                    framesresnext.append(frameno)
                    confidencelistresnext.append(confidence)
                vidcap.release()
                accuracy=round((sumval/frameno),2)
                print("* Total no of frames: ",frameno)
                print("* ResNext Estimated Accuracy: ",accuracy/100)

    input_path = r"D:\Xampp\htdocs\Main\upload"
    output_path = r"D:\Xampp\htdocs\Main\frame"
    num_frames = 200

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    framesmeso = []
    confidencelistmeso = []
    framenomeso =0

    to_iterate = list(os.walk(input_path))
    for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
        for file in files:
            if file.endswith('.mp4'):
                image_path = os.path.join(root, file)
                vidcap = cv2.VideoCapture(image_path)
                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Fixed number of frames (same interval between frames)
                frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int)

                out_path = image_path.replace(input_path, output_path)[:-4] + '/'  # Cut .mp4 suffix
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                success, image = vidcap.read()
                count = 0
                while success:
                    if count not in frame_idxs:
                        success, image = vidcap.read()
                        count += 1
                        continue
                    framenomeso +=1
                    cur_out_path = os.path.join(out_path, 'frame%d.jpg' % framenomeso)
                    cv2.imwrite(cur_out_path, image)  # save frame as JPEG file
                    framesmeso.append(framenomeso)
                    success, image = vidcap.read()
                    count += 1
                vidcap.release()
    # Instantiate a MesoNet model with pretrained weights
    meso = Meso4()
    meso.load('./weights/Meso4_DF')

    # Prepare image data
    # Rescaling pixel values (between 1 and 255) to a range between 0 and 1
    dataGenerator = ImageDataGenerator(rescale=1./255)

    # Instantiating generator to feed images through the network
    generator = dataGenerator.flow_from_directory(
        './temp_frame/',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Checking class assignment
    generator.class_indices

    # Recreate generator after removing '.ipynb_checkpoints'
    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
        './temp_frame/',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Re-check class assignment after removal
    generator.class_indices

    # Rendering image X with label y for MesoNet
    # X, y = generator.next()



    # Creating separate lists for correctly classified and misclassified images
    sumval=0
    l=[]
    upgradevalue=0
    count=0
    # Generating predictions on the validation set, storing in separate lists
    for i in range(len(generator.labels)):
        # Loading the next picture, generating a prediction
        X, y = generator.next()
        check=float(meso.predict(X)[0][0])
        upgradevalue=check*100
        confidencelistmeso.append(upgradevalue)
        if check<0.8:
            l.append(check)
        sumval=sumval+check
        count=count+1

        # Sorting into the proper category
        
    mesoaccuracy=round((sumval/count),2)
    print("* Mesonet Estimated Accuracy:",mesoaccuracy)

    overall_accuracy = accuracy + mesoaccuracy
    print()
    if (overall_accuracy) > 60.97:
        print("* Estimated Result: Video Might be Real")
    else:
        print("* Estimated Result: Video Might be Fake")
    
    # Plotting Resnext Confidence
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), gridspec_kw={'hspace': 0.5})

    axs[0].plot(framesresnext, confidencelistresnext, color='red')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Confidence')
    axs[0].set_title('Resnext Confidence of Each Frame')

    # Save the Resnext graph image
    resnext_graph_path = os.path.join('graph', 'resnext_confidence_graph.png')
    plt.savefig(resnext_graph_path)

    # Plotting MesoNet Confidence
    axs[1].plot(framesmeso, confidencelistmeso, color='blue')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Confidence')
    axs[1].set_title('MesoNet Confidence of Each Frame')

    # Save the MesoNet graph image
    meso_graph_path = os.path.join('graph', 'meso_confidence_graph.png')
    plt.savefig(meso_graph_path)

    # Plotting Both Resnext and MesoNet Confidence
    axs[2].plot(framesresnext, confidencelistresnext, color='red', label='Resnext')
    axs[2].plot(framesmeso, confidencelistmeso, color='blue', label='MesoNet')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Confidence')
    axs[2].set_title('Confidence of Each Frame')
    axs[2].legend()

    # Save the combined graph image
    combined_graph_path = os.path.join('graph', 'combined_confidence_graph.png')
    plt.savefig(combined_graph_path)

if __name__ == "__main__":
    main()
