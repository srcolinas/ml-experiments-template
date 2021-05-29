"""
In this module we store functions to measuer the performance of our model.

"""

from sklearn.metrics import mean_absolute_error, make_scorer


def get_metric_name_mapping():
    return {
        _mae(): mean_absolute_error,
        _cm(): custom_error
    }

def custom_error(y_true, y_pred):
    """A custom metric that is related to the business, the lower the better. """
    raise NotImplementedError


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _mae(): make_scorer(mean_absolute_error, greater_is_better=False, **params)
    }
    return mapping[name]


def _mae():
    return "mean absolute error"


def _cm():
    return "custom prediction error"

