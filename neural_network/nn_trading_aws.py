import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import datetime

# Define constants for learning rate and batch size
LEARNING_RATE = 0.001
BATCH_SIZE = 512
LOWER_THRESHOLD = 0.2
UPPER_THRESHOLD = 0.8

# Define a log directory (e.g., including a timestamp for organization)
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Read the data from the CSV file
file_path = '/home/ubuntu/desktop/data_nn.csv'
data = pd.read_csv(file_path)

# # Select first 1000 rows
# data = data.iloc[:20000]

# Separate features (X) and target (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #find the mean and std of the data
X_val = scaler.transform(X_val)     #use the mean and std of the training data to normalize the validation data
X_test = scaler.transform(X_test)  #use the mean and std of the training data to normalize the test data

# Determine the input shape
input_shape = (X_train.shape[1],)

# Define the model
model = keras.Sequential([
    # layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(512, activation='relu',input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(1, activation='sigmoid'),
])

# Print the model summary
# model.summary()

# Compile the model
model.compile(
    # optimizer='adam',
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
    loss='binary_crossentropy',
    metrics=['binary_accuracy']    
)

# Define the callbacks
early_stopping = EarlyStopping(
    patience=10,    # number of epochs to wait before early stopping
    min_delta=0.001,   # change in validation loss to qualify as improvement
    restore_best_weights=True,
)

# Create the TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,   # Log histograms of weights after each epoch
    write_graph=True,   # Visualize the computational graph
    write_images=True   # Log model weights as images
)

# Create the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='/home/ubuntu/desktop/model_checkpoint.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=1000, 
    callbacks=[early_stopping, tensorboard_callback, checkpoint_callback]
    )

# Plot the training and validation loss
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
# Show the plots
plt.show()

# Save the model
model.save('/home/ubuntu/desktop/model.keras')


################################################################################
################################################################################
################################################################################


# load model_checkpoint.keras and predict on X_test

# Load the model
model = keras.models.load_model('/home/ubuntu/desktop/model_checkpoint.keras')

# Predict on the test data
y_pred = model.predict(X_test)
print("Predictions vs Actuals:")
for pred, actual in zip(y_pred[-20:], y_test[-20:]):
    print(f'Predicted: {pred}, Actual: {actual}')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

import matplotlib.pyplot as plt
plt.hist(y_pred, bins=50)
plt.title('Prediction Distribution')
plt.show()

# Generate predictions
y_pred_class = []
for pred in y_pred:
    if pred < LOWER_THRESHOLD:
        y_pred_class.append(0)  # Confidently classify as 0
    elif pred > UPPER_THRESHOLD:
        y_pred_class.append(1)  # Confidently classify as 1
    else:
        y_pred_class.append(-1)  # Leave undecided

# Evaluate the confident predictions
filtered_actuals = [actual for pred, actual in zip(y_pred_class, y_test) if pred != -1]
filtered_preds = [pred for pred in y_pred_class if pred != -1]

# Use sklearn metrics to evaluate
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print("Confident Predictions Metrics:")
print(f"Precision: {precision_score(filtered_actuals, filtered_preds):.4f}")
print(f"Recall: {recall_score(filtered_actuals, filtered_preds):.4f}")
print(f"Accuracy: {accuracy_score(filtered_actuals, filtered_preds):.4f}")
print(f"F1 Score: {f1_score(filtered_actuals, filtered_preds):.4f}")

#print length of test data
print(len(y_test))
print(len(y_pred_class))
print(len(filtered_actuals))  
print(len(filtered_preds))
