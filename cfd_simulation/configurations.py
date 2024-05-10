import os
import numpy as np
import json
from constants import NUM_SEQUENCES, OUTPUTS_FOLDER, SIM_CONFIGURATIONS, WIDTH, HEIGHT
from sklearn.model_selection import ParameterSampler


def _select_parameters(amount, param_grid):
    return list(ParameterSampler(param_grid, n_iter=amount, random_state=42))

def _build_configuration(id, obstacle_type, obstacle_parameters):
    return {
        "id": id,
        "obstacle": {
            "type": obstacle_type,
            "parameters": obstacle_parameters
        }
    }

def _ellipse_configurations(configurations, id, amount_ellipses):
    ellipses_param_grid = {
        "center_x": range(WIDTH//4, WIDTH//2, 10),
        "center_y": range(HEIGHT//3, 2*HEIGHT//3, 10),
        "semi_major_axis": range(HEIGHT//5, HEIGHT//3, 5),
        "semi_minor_axis_proportion": np.arange(1/5, 1/4, 0.01),
        "degrees": range(-30, 30, 10)
    }
    ellipses_parameters = _select_parameters(amount_ellipses, ellipses_param_grid)
    for parameters in ellipses_parameters:
        configurations.append(
            _build_configuration(id, "ellipse", parameters)
        )
        id+=1

def _circumference_configurations(configurations, id, amount_circumference):
    circumference_param_grid = {
        "center_x": range(WIDTH//4, WIDTH//2, 10),
        "center_y": range(HEIGHT//3, 2*HEIGHT//3, 10),
        "radius": range(HEIGHT//9, HEIGHT//5, 3)
    }
    circumference_parameters = _select_parameters(amount_circumference, 
                                                  circumference_param_grid)
    for parameters in circumference_parameters:
        configurations.append(
            _build_configuration(id, "circumference", parameters)
        )
        id+=1

def create_configurations():
    configurations = []
    id = 0
    
    shapes_number = 2
    amount, extra = divmod(NUM_SEQUENCES, shapes_number)
    amount_circumference = amount
    amount_ellipses = amount + extra

    _circumference_configurations(configurations, id, amount_circumference)
    
    _ellipse_configurations(configurations, id, amount_ellipses)

    try:
        # create folder to store results
        os.makedirs(OUTPUTS_FOLDER)
    except OSError:
        # folder already exists
        pass

    with open(SIM_CONFIGURATIONS, 'w') as f:
        json.dump(configurations, f)
