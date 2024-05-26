#Diseño del modelo de prediccion de diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import keras
import numpy as np

from TF2JSON.TF2JSON import ConvertTF2JSON

model = Sequential()

model.add(Flatten(input_shape=(784, 1)))

model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(10, activation='softmax'))

# compiling the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Saving the model
model_name = "nist_fcl"
model.save("./" + model_name + ".keras")

# Load the model
#model = keras.models.load_model("./nist_fcl.keras")
# Print the model summary
model.summary()

#key words
kwargs = {}
kwargs["model"] = model
kwargs["model_info"] = {
    "nombre": "NIST Fully connected layer based model",
    "version": "v1.0.0",
    "owner": "Tekvot"
}
kwargs["model_file_name"] = model_name + ".json"

kwargs = ConvertTF2JSON.run(**kwargs)

