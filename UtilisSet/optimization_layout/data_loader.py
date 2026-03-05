from pathlib import Path
import os
import UtilisSet.utils as utils
import torch
def run_model_in_testing_event(saved_objects_dir):
    saved_simulation_path = Path(saved_objects_dir) / "saved_normalized_testing_windows"
    list_of_simulations = os.listdir(saved_simulation_path)
    saved_normalizers_path = Path(saved_objects_dir) / "saved_normalizers"

    normalizer = utils.load_pickle(
        saved_normalizers_path
        / 'normalizer_development_tuindorp.pk'
    )
    true_values=[]
    for event in list_of_simulations:
        norm_in_window = utils.load_pickle(
            saved_simulation_path
            / event
        )
        target_heads = torch.stack([real_y["y"] for real_y in norm_in_window], dim=0)
        swmm_heads_pd = normalizer.get_unnormalized_heads_pd(target_heads)
        true_values.append(swmm_heads_pd)
    return true_values
