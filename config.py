import numpy as np
width = 128
height = 64
initconfig = {
    'DOF_PN': 3,
    'ELEM_K': 'Q4',
    'ETA': '0.5',
    'FILT_RAD': 2,
    'NUM_ELEM_Z': 0,
    'NUM_ELEM_X': width,
    'NUM_ELEM_Y': height,
    'PROB_TYPE': 'comp',
    'NUM_ITER': 40,
    'P_FAC': 2.0,
    'Q_FAC': 1,
    # Grey-scale filter (GSF)
    'P_HOLD '    : 5 , # num of iters to hold p constant from start
    'P_INCR '    : 0.25 , # increment by this amount
    'P_CON'      : 1 , # increment every 'P_CON' iters
    'P_MAX '     : 3 , # max value of 'P_CON'
    'Q_HOLD'     : 10 , # num of iters to hold q constant from start
    'Q_INCR'     : 0.5 , # increment by this amount
    'Q_CON'      : 5 , # increment every 'Q_CON' iters
    'Q_MAX'      : 5,  # max value of 'Q_CON'
    }


def choose_nodes(n):
    node_shape = (height + 1, width + 1)
    node_max = (height + 1)*(width + 1)
    node_args = np.arange(1, node_max + 1)
    probability = np.full(shape=node_shape, fill_value= 500.0)
    probability[1:-2, 1:-2] = 1.0
    probability /= probability.sum()
    probs = probability.T.ravel()
    nodes = np.random.choice(node_args, size=n,replace=False, p=probs)
    return nodes


def random_config():
    fxtr_x = np.random.randint(1, 6)
    fxtr_y = np.random.randint(1, 6)
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)
    config =initconfig.copy()
    config['FXTR_NODE_X'] = choose_nodes(fxtr_x)
    config['FXTR_NODE_Y'] = choose_nodes(fxtr_y)

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [-1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.3)

    return config


def cantilever_beam_config():

    config =initconfig.copy()
    fxtr_x = np.array([33,1])
    fxtr_y = np.array(list(range(1, 33)))
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)

    config['FXTR_NODE_X'] = fxtr_x
    config['FXTR_NODE_Y'] = fxtr_y

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.3)

    return config


def simply_supported_beam_config():
    config =initconfig.copy()

    fxtr_y = np.array([2112,2145])
    fxtr_x = np.array([32,33])
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)

    config['FXTR_NODE_X'] = fxtr_x
    config['FXTR_NODE_Y'] = fxtr_y

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.1)

    return config


def continuous_beam_config():
    config =initconfig.copy()

    fxtr_y = np.array([33,2145])
    fxtr_x = np.array([33,1089,1584,2145])
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)

    config['FXTR_NODE_X'] = fxtr_x
    config['FXTR_NODE_Y'] = fxtr_y

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.1)

    return config


def random_config_64_128():
    fxtr_x = np.random.randint(1, 6)
    fxtr_y = np.random.randint(1, 6)
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)
    config =initconfig.copy()
    config['FXTR_NODE_X'] = choose_nodes(fxtr_x)
    config['FXTR_NODE_Y'] = choose_nodes(fxtr_y)

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [-1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.1)

    return config


def cantilever_beam_config_64_128():

    config =initconfig.copy()
    fxtr_x = np.array([65,1])
    fxtr_y = np.array(list(range(1, 65)))
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)

    config['FXTR_NODE_X'] = fxtr_x
    config['FXTR_NODE_Y'] = fxtr_y

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.1)

    return config


def simply_supported_beam_config_64_128():
    config =initconfig.copy()

    fxtr_y = np.array([8128,8192])
    fxtr_x = np.array([64,65])
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)

    config['FXTR_NODE_X'] = fxtr_x
    config['FXTR_NODE_Y'] = fxtr_y

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.1)

    return config


def continuous_beam_config_64_128():
    config =initconfig.copy()

    fxtr_y = np.array([64,8192])
    fxtr_x = np.array([64,7232,7296,4544,4608])
    load_x = np.random.randint(1, 10)
    load_y = np.random.randint(1, 10)

    config['FXTR_NODE_X'] = fxtr_x
    config['FXTR_NODE_Y'] = fxtr_y

    config['LOAD_NODE_X'] = choose_nodes(load_x)
    config['LOAD_VALU_X'] = [1] * load_x

    config['LOAD_NODE_Y'] = choose_nodes(load_y)
    config['LOAD_VALU_Y'] = [-1] * load_y
    config['VOL_FRAC'] = np.random.normal(0.8, 0.1)

    return config