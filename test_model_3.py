""" First TF model to test. It's based on fully connected layers
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from TF2JSON.TF2JSON import ConvertTF2JSON

# Create a Sequential model
model = Sequential()

# Add a Flatten layer to flatten the input image
model.add(Flatten(input_shape=(5,1)))

# Add two dense layers with 200 units and 'relu' activation function
model.add(Dense(4, activation='relu'))

# Add a softmax output layer with 10 units
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

# Print the model summary
model.summary()

#key words
kwargs = {}
kwargs["model"] = model
kwargs["model_info"] = {
    "nombre": "Test Dense based Model 3",
    "version": "v1.0.0",
    "owner": "Tekvot"
}
kwargs["model_file_name"] = "test_dense_model_3.json"

kwargs = ConvertTF2JSON.run(**kwargs)



