
from UtilisSet.Reconstruction_models.MLP_model import *
from UtilisSet.Reconstruction_models.GNN_model import *
from UtilisSet.Reconstruction_models.Transformer_CNN import *
from UtilisSet.Reconstruction_models.Transformer_GINEConv import *
def ModelFactory(model_name):
    available_models = {
        "MLP_Benchmark_metamodel": MLP_Benchmark_metamodel,
        "NN_GINEConv_NN": NN_GINEConv_NN,
        "TransferModel": TransferModel,
        "FusedCNNTransformer":FusedCNNTransformer,
        "Transformer_GINEConv_NN":Transformer_GINEConv_NN
    }
    model = available_models[model_name]
    return model
