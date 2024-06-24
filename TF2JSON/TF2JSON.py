import os
from abc import ABC, abstractmethod
import json

import numpy as np

verbose = False

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
        print("Get AI model components from keras file")
        W, B, A, N = [], [], [], []
        tP, tL, tN = 0, 0, 0
        model = kwargs["model"]

        for i in range(1, len(model.layers)):
            weights, biases = model.layers[i].get_weights()
            tP += len(weights)
            W.append(weights)
            B.append(biases)
            A.append(str(model.layers[i].activation).split(" ")[1])
            print(f"Layer {i} loaded")

        if verbose:
            print(f"W: {W}")
            print(f"B: {B}")
            print(f"A: {A}")

        for i, ws in enumerate(W):
            if verbose: print(ws[0,:])
            n = []
            #print(f"len ws: {len(ws)}")
            #print(f"len wsi: {len(ws[i])}")
            for j in range(len(ws[0])):
                n.append(ws[:,j])
            tN += len(n)
            N.append(n)
        if verbose: print(N)
        print(f"Numero de capas: {tN}") #Numero de capas
        tL = len(N)

        kwargs["W"], kwargs["B"], kwargs["A"], kwargs["N"] = W, B, A, N
        #total Parameters, layers and neurons
        kwargs["tP"], kwargs["tL"], kwargs["tN"] = tP, tL, tN

        return kwargs

class complete_model_info(convertion_ops):
    """ add model metainformation
    """

    def run_operation(**kwargs):
        print("Completing model information")
        kwargs["model_info"]["neurons_num"] = kwargs["tN"]
        kwargs["model_info"]["layers_num"] = kwargs["tL"]
        kwargs["model_info"]["params_num"] = kwargs["tP"]

        return kwargs

class get_JSON_string(convertion_ops):
    """ add W, B, A and N to a JSON string
    """

    def run_operation(**kwargs):
        print("Getting JSON String")

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
          print(f"---- {nm} ----")
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
                  #"inputs_names": ["x"+str(i) for i in range(1, n_ent+1)],
                  "i": ["x" + str(1), "x" + str(n_ent)], #input names
                  "o": [on], #output names
                  "p": { #weights
                      "w": [round(float(N[c][n][i]), 1) for i in range(len(N[c][n]))]
                  },
                  "b": float(B[c][n]), #bias
                  "f": A[c] #activation function
              }
            else:
              model_dict["layers"][nm][nnm] = {
                  "i": [oss[0], oss[-1]], #oss.copy(),
                  "o": [on],
                  "p": {
                      "w": [round(float(N[c][n][i]), 1) for i in range(len(N[c][n]))]
                  },
                  "b": float(B[c][n]),
                  "f": A[c]
              }

        if verbose: print(model_dict)
        kwargs["model_dict"] = model_dict

        return kwargs


class JSONobj2File(convertion_ops):
    """ get a JSON file from the obj. Save it in same project folder
    """

    def run_operation(**kwargs):
        try:
            smj = kwargs["save_model_json"]
        except:
            smj = False
        if smj:
            print("JSON obj to file")
            model_file_name = kwargs["model_file_name"]
            model_dict = kwargs["model_dict"]
            with open(model_file_name, "w") as outfile:
                json.dump(model_dict, outfile)
            #Display the json string that was saved
            print("Json file saved")
            if verbose:
                json_object = json.dumps(model_dict, indent = 4)
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

