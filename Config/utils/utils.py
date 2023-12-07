def unify_args(low_prior_dict, high_prior_dict):
    """
    :param low_prior_dict: low priority dict
    :param high_prior_dict: high priority dict
    :return: unified dictionary, which high priority dict overrides the low priority dict
    """
    unified_dict = low_prior_dict
    for key in high_prior_dict.keys():
        unified_dict[key] = high_prior_dict[key]
    return unified_dict

def unify_known_args(low_prior_dict, high_prior_dict):
    """
    if high_prior_dict has an element which is not None, we use that element,
    else we keep the value coming from low_prior_dicts
    :param low_prior_dict: low priority dict
    :param high_prior_dict: high priority dict
    :return: unified dictionary, which high priority dict overrides the low priority dict
    """
    unified_dict = low_prior_dict
    for key in high_prior_dict.keys():
        if not high_prior_dict[key] is None:
            unified_dict[key] = high_prior_dict[key]
    return unified_dict


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

if __name__ == "__main__":
    dict1 = {
            "1" : "value11",
            "2" : "value12",
            "3" : "value13",
            "4" : "value14"}
    dict2 = {
            "2" : "value22",
            "3" : "value23",
            "4" : "value24",
            "5" : "value25"}

    print(unify_args(dict1, dict2))

