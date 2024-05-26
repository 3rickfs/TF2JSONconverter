""" First TF model to test. It's based on fully connected layers
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from TF2JSON.TF2JSON import ConvertTF2JSON

# Create a Sequential model
model = Sequential()

model.add(Flatten(input_shape=(3,1)))
model.add(Dense(3, activation='relu'))

model.add(Dense(2, activation='relu'))

# Print the model summary
model.summary()

#save the model
model_name = "test_dense_model_5"
model.save("./" + model_name + ".keras")

#key words
kwargs = {}
kwargs["model"] = model
kwargs["model_info"] = {
    "nombre": "Test Dense based Model 5",
    "version": "v1.0.0",
    "owner": "Tekvot"
}
kwargs["model_file_name"] = "test_dense_model_5.json"

kwargs = ConvertTF2JSON.run(**kwargs)



