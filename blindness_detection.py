import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
import os


base_path = "C:/Users/gyash/OneDrive/Desktop/aptos_project" 

csv_path = os.path.join(base_path, "train.csv")
train_images_dir = os.path.join(base_path, "train_images")

tf.random.set_seed(42)
np.random.seed(42)

print("Loading data...")
df = pd.read_csv(csv_path)
df.columns = ['id_code', 'diagnosis']
df['id_code'] = df['id_code'].astype(str) + '.png'

print("\nClass Distribution:")
print(df['diagnosis'].value_counts().sort_index())
plt.figure(figsize=(10, 5))
sns.countplot(x=df['diagnosis'])
plt.title('Class Distribution')
plt.savefig(os.path.join(base_path, 'class_distribution.png'))
plt.show()

class_weights = compute_class_weight('balanced',
                                     classes=np.unique(df['diagnosis']),
                                     y=df['diagnosis'])
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

print("Setting up data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_images_dir,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    color_mode='rgb',
    shuffle=True
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=train_images_dir,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    color_mode='rgb',
    shuffle=False
)

print(f"Found {train_generator.n} training images.")
print(f"Found {validation_generator.n} validation images.")
print("Data preparation complete!")

print("\nBuilding the model...")

for i in range(1):
    images, labels = next(train_generator)
    print(f"Actual image shape from generator: {images[0].shape}")
    break

from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model built successfully!")
model.summary()

print("\nStarting training...")
checkpoint = ModelCheckpoint(os.path.join(base_path, 'best_model.h5'),
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=2,
                              min_lr=1e-7,
                              verbose=1)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=5,
                           restore_best_weights=True,
                           verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[checkpoint, reduce_lr, early_stop],
    class_weight=class_weights,
    verbose=1
)

print("\nEvaluating model...")
best_model = tf.keras.models.load_model(os.path.join(base_path, 'best_model.h5'))

validation_generator.reset()
y_true = validation_generator.labels 

preds = best_model.predict(validation_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f"\nQuadratic Weighted Kappa: {kappa:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['0', '1', '2', '3', '4']))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2', '3', '4'], 
            yticklabels=['0', '1', '2', '3', '4'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_path, 'confusion_matrix.png'))
plt.show()

print("Training and evaluation complete!")