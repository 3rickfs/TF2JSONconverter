""" Forth TF model to test. It's based on fully connected layers
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from TF2JSON.TF2JSON import ConvertTF2JSON

# Create a Sequential model
model = Sequential()

# Add a Flatten layer to flatten the input image
model.add(Flatten(input_shape=(6,1)))

model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

# Print the model summary
model.summary()

#key words
kwargs = {}
kwargs["model"] = model
kwargs["model_info"] = {
    "nombre": "Test Dense based Model 4",
    "version": "v1.0.0",
    "owner": "Tekvot"
}
kwargs["model_file_name"] = "test_dense_model_4.json"

kwargs = ConvertTF2JSON.run(**kwargs)



