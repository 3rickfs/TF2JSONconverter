import os
from abc import ABC, abstractmethod
import json

import numpy as np

class convertion_ops(ABC):
    """ Abstract class to be inherited from ops childs to convert from
        TF to JSON
    """
    @abstractmethod
    def run_operation(**kwargs):
        #interface to child classes
        pass


class get_AI_model_components(convertion_ops):
    """ Components like weights, biases, activation functions, etc.
    """

    def run_operation(**kwargs):
        W, B, A, N = [], [], [], []
        model = kwargs["model"]

        for i in range(1, len(model.layers)):
          weights, biases = model.layers[i].get_weights()
          W.append(weights)
          B.append(biases)
          A.append(str(model.layers[i].activation).split(" ")[1])

        print(f"W: {W}")
        print(f"B: {B}")
        print(f"A: {A}")

        for i, ws in enumerate(W):
          print(ws[0,:])
          n = []
          #print(f"len ws: {len(ws)}")
          #print(f"len wsi: {len(ws[i])}")
          for j in range(len(ws[0])):
            n.append(ws[:,j])
          N.append(n)
        print(N)
        print(f"Numero de capas: {len(N)}") #Numero de capas

        kwargs["W"], kwargs["B"], kwargs["A"], kwargs["N"] = W, B, A, N

        return kwargs

class get_JSON_string(convertion_ops):
    """ add W, B, A and N to a JSON string
    """

    def run_operation(**kwargs):

        W, B, A, N = kwargs["W"], kwargs["B"], kwargs["A"], kwargs["N"]

        model_dict = {}
        model_dict["model_info"] = kwargs["model_info"]
        model = kwargs["model"]

        n_ent = model.layers[0].get_config()['batch_input_shape'][1]
        model_dict["input_layer"] = {
            "n_entradas": n_ent
        }

        model_dict["layers"] = {}
        nc = 0 #neuron units counter
        ons = [] #outputs names
        for c in range(len(N)):
          nm = "layer_" + str(c+1)
          model_dict["layers"][nm] = {}
          oss = ons.copy()
          ons = []
          for n in range(len(N[c])):
            nnm = "neuron_" + str(nc+1)
            on = "o"+str(nc+1) #nc to count the outputs as well
            nc += 1
            ons.append(on)
            if c < 1:
              model_dict["layers"][nm][nnm] = {
                  "inputs_names": ["x"+str(i) for i in range(1, n_ent+1)],
                  "outputs_names": [on],
                  "pesos": {
                      "w": [float(N[c][n][i]) for i in range(len(N[c][n]))]
                  },
                  "bias": float(B[c][n]),
                  "fa": A[c]
              }
            else:
              model_dict["layers"][nm][nnm] = {
                  "inputs_names": oss.copy(),
                  "outputs_names": [on],
                  "pesos": {
                      "w": [float(N[c][n][i]) for i in range(len(N[c][n]))]
                  },
                  "bias": float(B[c][n]),
                  "fa": A[c]
              }

        print(model_dict)
        kwargs["model_dict"] = model_dict

        return kwargs


class JSONobj2File(convertion_ops):
    """ get a JSON file from the obj. Save it in same project folder
    """

    def run_operation(**kwargs):
        model_file_name = kwargs["model_file_name"]
        model_dict = kwargs["model_dict"]
        with open(model_file_name, "w") as outfile:
            json.dump(model_dict, outfile)
        #Display the json string that was saved
        json_object = json.dumps(model_dict, indent = 4)
        print("Json file saved:")
        print(json_object)

        return kwargs

class ConvertTF2JSON:
    """ Call this class to run above coded operations
    """

    @staticmethod
    def run(**kwargs):
        for operation in convertion_ops.__subclasses__():
            kwargs = operation.run_operation(**kwargs)
        return kwargs

