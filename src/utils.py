import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer


def load_data(data_type="classification"):
    """
    [summary]

    Parameters
    ----------
    data_type : str, optional
        [description], by default "classification"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    data_type = data_type.lower()
    error_message = (
        "No data_type was specified, use either 'classification' or 'regression'"
    )
    assert data_type in ["classification", "regression"], error_message

    if data_type == "classification":
        cancer = load_breast_cancer()
        x = pd.DataFrame(
            np.c_[cancer["data"], cancer["target"]],
            columns=np.append(cancer["feature_names"], ["target"]),
        )
        y = x.pop("target")
    else:
        boston = load_boston()
        x = pd.DataFrame(
            np.c_[boston["data"], boston["target"]],
            columns=np.append(boston["feature_names"], ["target"]),
        )
        y = x.pop("target")
    return x, y
