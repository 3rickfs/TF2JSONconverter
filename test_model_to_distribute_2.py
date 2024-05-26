#Diseño del modelo de prediccion de diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import keras
import numpy as np

from TF2JSON.TF2JSON import ConvertTF2JSON


# Create a Sequential model
model = Sequential()

# Add a Flatten layer to flatten the input image
model.add(Flatten(input_shape=(5, 1)))

# Add three dense layers with 'relu' activation function
model.add(Dense(10, activation='relu'))

# Add a softmax output layer with 2 units
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='relu'))

# compiling the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Print the model summary
model.summary()

#save the model
model_name = "model_to_be_distributed_and_executed_2"
model.save("./" + model_name + ".keras")

#key words
kwargs = {}
kwargs["model"] = model
kwargs["model_info"] = {
    "nombre": "Model to be distributed and run by NODs",
    "version": "v1.0.0",
    "owner": "Tekvot"
}
kwargs["model_file_name"] = model_name + ".json"

kwargs = ConvertTF2JSON.run(**kwargs)

#Running model
model_input = [2, 3, 4, 5, 8]
inp = np.array(model_input).reshape(1, 5)
pred = model.predict(inp)

print(f"pred: {pred}")

