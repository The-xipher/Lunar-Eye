import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
from tqdm import tqdm

# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

# Create a Classifier class
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
    
input_path = r"D:\Xampp\htdocs\Main\upload"
output_path = r"D:\Xampp\htdocs\Main\frame"
num_frames = 200

if not os.path.exists(output_path):
    os.makedirs(output_path)
frame=[]
frameno=0
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
                frameno += 1
                cur_out_path = os.path.join(out_path, 'frame%d.jpg' % frameno)
                cv2.imwrite(cur_out_path, image)  # save frame as JPEG file
                frame.append(frameno)
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
    './frame/',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')

# Checking class assignment
generator.class_indices

# Recreate generator after removing '.ipynb_checkpoints'
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
    './frame/',
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
confidence=[]
count=0
# Generating predictions on the validation set, storing in separate lists
for i in range(len(generator.labels)):
    # Loading the next picture, generating a prediction
    X, y = generator.next()
    check=float(meso.predict(X)[0][0])
    confidence.append(check)
    if check<0.8:
        l.append(check)
    sumval=sumval+check
    count=count+1

mesoaccuracy=round((sumval/count),2)
    # Sorting into the proper category
    

    # Printing status update
print("* Total No of Frames:",count)
print("* Estimated Accuracy:",mesoaccuracy)

if (mesoaccuracy)>0.97 and len(l)<2:
    print("* Estimated Result: Video Might be Real")
else:
    print("* Estimated Result: Video Might be Fake")

# Plotting the graph
plt.plot(frame, confidence, color='red')
plt.xlabel('Frame')
plt.ylabel('Confidence')
plt.title('Confidence of Each Frame')

# Save the graph image to the 'graph' folder
graph_folder = './graph'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)
graph_path = os.path.join(graph_folder, 'confidence_graph.png')
plt.savefig(graph_path)
