
from UtilisSet.Reconstruction_models.Proposed_Model import *
def ModelFactory(model_name):
    available_models = {
        "Edge-STGCN":Edge_STGCN
    }
    model = available_models[model_name]
    return model
