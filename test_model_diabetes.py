#Diseño del modelo de prediccion de diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from TF2JSON.TF2JSON import ConvertTF2JSON


# Create a Sequential model
model = Sequential()

# Add a Flatten layer to flatten the input image
model.add(Flatten(input_shape=(31, 1)))

# Add three dense layers with 'relu' activation function
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(120, activation='relu'))

# Add a softmax output layer with 2 units
model.add(Dense(2, activation='softmax'))

# compiling the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Print the model summary
model.summary()

#save the model
model_name = "diabetes_detection_model"
model.save("./" + model_name + ".keras")


#key words
kwargs = {}
kwargs["model"] = model
kwargs["model_info"] = {
    "nombre": "Diabeter detection model",
    "model_version": "v1.0.0"
}
kwargs["model_file_name"] = model_name + ".json"

kwargs = ConvertTF2JSON.run(**kwargs)

#Running model
model_input = [1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,
               1,0,0,1,1,0,0,0,1,1,1,1,1,0,0
              ]

inp = np.array(model_input).reshape(1, 31)
pred = model.predict(inp)

print(f"pred: {pred}")


