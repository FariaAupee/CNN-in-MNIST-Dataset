import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import sys

# Check if an argument is provided
if len(sys.argv) > 1:
    # Access the argument from the command line and store it in a variable
    path = sys.argv[1]
    print("Path provided on the command line:", path)
else:
    print("No path provided. Please provide a path as a command-line argument.")


#Data Analysis Part

# Load the training and testing data
train_data = pd.read_csv(path+'/fashion-mnist_train.csv')
test_data = pd.read_csv(path+'/fashion-mnist_test.csv')

# Display the first few rows of the data to verify the loading
print("Training Data:")
print(train_data.head())
print("\nTesting Data:")
print(test_data.head())
#number of samples in train and test data
num_train_sample = train_data.shape[0]
num_test_sample = test_data.shape[0]

#dimension of each image
image_dim = train_data.iloc[0,1:].shape

#number of classes
num_classes = train_data['label'].nunique()

print(num_train_sample)
print(num_test_sample)
print(image_dim)
print(num_classes)

# Visualize the distribution of classes
class_counts = train_data['label'].value_counts().sort_index()
class_labels = class_counts.index

plt.figure(figsize=(10, 5))
plt.bar(class_labels, class_counts)
plt.title("Class Distribution in the Training Set")
plt.xlabel("Class Label")
plt.ylabel("Number of Samples")
plt.xticks(class_labels)
plt.show()

# Visualize example images from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle("Example Images from Each Class")

for i, ax in enumerate(axes.flatten()):
    class_samples = train_data[train_data['label'] == i]
    sample_image = class_samples.iloc[0, 1:].values.reshape(28, 28)
    ax.imshow(sample_image, cmap='gray')
    ax.set_title(f"Class {i}")
    ax.axis('off')

plt.show()

# Analyze feature statistics for different classes
class_features = []

# Iterate through each class label (0 to 9)
for label in range(10):
    # Select samples of the current class
    class_samples = train_data[train_data['label'] == label]

    # Compute mean values for each feature (pixel) in the class
    class_mean = class_samples.iloc[:, 1:].mean()

    # Store the mean values and class label
    class_features.append((label, class_mean))

# Plot the mean feature values for each class
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle("Mean Feature Values for Each Class")

for i, (label, mean_features) in enumerate(class_features):
    ax = axes[i // 5, i % 5]
    ax.imshow(np.array(mean_features).reshape(28, 28), cmap='gray')
    ax.set_title(f"Class {label}")
    ax.axis('off')

plt.show()

# Separate the labels from the features
X = train_data.iloc[:, 1:]  # Features (pixel values)
y = train_data['label']     # Labels (class labels)

# Standardize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce the dimensionality
n_components = 2  # Number of components to keep (you can change this)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'PCA of Fashion MNIST (Reduced to {n_components} Dimensions)')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()


#Train Model Part

# Prepare the data
X = train_data.iloc[:, 1:].values
y = train_data['label'].values

# Normalize pixel values to a range of 0 to 1
X = X / 255.0

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for CNN (28x28 pixels with 1 channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

# One-hot encode the class labels
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

# Build a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Visualize training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()


#Test Part

# Load and preprocess the test data (similar to training and validation data)
X_test = test_data.iloc[:, 1:].values
y_test = test_data['label'].values
X_test = X_test / 255.0  # Normalize pixel values
X_test = X_test.reshape(-1, 28, 28, 1)  # Reshape for CNN
y_test = to_categorical(y_test, num_classes=10)  # One-hot encode labels

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Prediction on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

class_names = ['T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot']

report = classification_report(y_true,y_pred_classes, target_names= class_names)
print(report)

# Generate outputs.txt file
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))

# Save the model summary to a text file
with open('outputs.txt', 'w') as f:
    f.write('\n'.join(model_summary))
    f.write('\n\nClassification Report:\n')
    f.write(report)

print("Model summary saved to 'outputs.txt'.")