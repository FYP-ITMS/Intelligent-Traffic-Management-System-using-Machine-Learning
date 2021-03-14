def parse_model_configuration(path):
    """
    Function:
        Takes a configuration file
    Arguments:
        path -- path of model configuration file
        
    Returns:
        blocks -- a list of blocks. Each blocks describes a block in the neural
                network to be built. Block is represented as a dictionary in the list
    """
    with open(path) as cfg:
        lines = cfg.read()
    lines = lines.split("\n")
    lines = [line for line in lines if not line.startswith("#") and line]
    lines = [line.strip() for line in lines]
    blocks = []
    for line in lines:
        if line.startswith("["):
            blocks.append({})
            blocks[-1]["type"] = line[1:-1].strip()
            if blocks[-1]["type"] == "convolutional":
                blocks[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            key, value = key.strip(), value.strip()
            blocks[-1][key] = value
    return blocks

def parse_data_config(path):
    """
    Function:
        Parses the data configuration file
    
    Arguments:
        path -- path of data configuration file
        
    Returns:
        options -- dictionary of data configurations options
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def load_classes(path):
    """
    Function:
        Loads class labels at 'path'
    Arguments:
        path -- path of class labels file   
    Returns:
        name -- a list of class labels
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names
