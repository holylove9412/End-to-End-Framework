from pathlib import Path
from dotenv import load_dotenv
from UtilisSet.utils import load_yaml
from UtilisSet.Backbone import Backbone

load_dotenv()
swmm_executable_path = 'D:/EPASWMM5.1.015/swmm5.exe'


if __name__ == '__main__':
    current_file = Path(__file__)
    current_dir = current_file.parent
    swmm_data_folder = current_dir/'data'/'SWMM_data'
    results_data_folder = current_dir/'data'/'Results_data'

    yaml_folder = current_dir/'configs'
    yaml_name = "study_area.yaml"
    project_name = 'Optimal layout'

    yaml_path = yaml_folder / yaml_name

    yaml_data = load_yaml(yaml_path)
    inp_file = current_dir/'network'/yaml_data["network"]/(yaml_data["network"]+'.inp')

    config = yaml_data

    ml_experiment = Backbone(config, swmm_data_folder, results_data_folder)
    #
    # ml_experiment.train_model()

    # ml_experiment.get_testing_loader()
    # ml_experiment.run_model_in_validation_event(event_index=8)
    ml_experiment.run_model_in_testing_event()
    #
    # ml_experiment.display_results()

