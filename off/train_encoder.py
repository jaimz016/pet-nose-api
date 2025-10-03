# train_encoder.py (minimal)
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

DATA_DIR = "data"   # from step 1
IMG_SIZE = (224,224)
BATCH = 16
EMBED_DIM = 128

# use ImageDataGenerator to load per-class folders
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1,
                               rotation_range=15, brightness_range=(0.8,1.2))
train_flow = train_gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                           batch_size=BATCH, class_mode='categorical', subset='training')
val_flow = train_gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                         batch_size=BATCH, class_mode='categorical', subset='validation')

num_classes = train_flow.num_classes

base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, pooling='avg')
x = base.output
x = layers.Dense(EMBED_DIM, activation=None, name="embedding")(x)   # embedding layer (no activation)
# add normalization (optional)
x_norm = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2norm")(x)
logits = layers.Dense(num_classes, activation='softmax', name="logits")(x_norm)

model = Model(inputs=base.input, outputs=logits)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_flow, validation_data=val_flow, epochs=12)

# create encoder model that outputs embedding vector
encoder = Model(inputs=base.input, outputs=model.get_layer("embedding").output)
encoder.save("encoder.h5")
