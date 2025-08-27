# Copyright (c) Microsoft Corporation.
# Copyright (c) Chester Luo
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from typing import Union, List, Type
from scipy.stats import percentileofscore
from qlib.data.ops import NpElemOperator, NpPairOperator, ElemOperator
from qlib.data.ops import OpsList

np.seterr(invalid="ignore")


class Sqrt(NpElemOperator):
    """Feature Sqrt

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with sqrt
    """

    def __init__(self, feature):
        super(Sqrt, self).__init__(feature, "sqrt")


class Exp(NpElemOperator):
    """Feature Exp

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with exp
    """

    def __init__(self, feature):
        super(Exp, self).__init__(feature, "exp")


class Square(NpElemOperator):
    """Feature Square

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with square
    """

    def __init__(self, feature):
        super(Square, self).__init__(feature, "square")


class Sin(NpElemOperator):
    """Feature Sin

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with sin
    """

    def __init__(self, feature):
        super(Sin, self).__init__(feature, "sin")


class Cos(NpElemOperator):
    """Feature Cos

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with cos
    """

    def __init__(self, feature):
        super(Cos, self).__init__(feature, "cos")


class Tan(NpElemOperator):
    """Feature Tan

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with tan
    """

    def __init__(self, feature):
        super(Tan, self).__init__(feature, "tan")


class Tanh(NpElemOperator):
    """Feature Tanh

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with tanh
    """

    def __init__(self, feature):
        super(Tan, self).__init__(feature, "tanh")


class Arcsin(NpElemOperator):
    """Feature Arcsin

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with arcsin
    """

    def __init__(self, feature):
        super(Arcsin, self).__init__(feature, "arcsin")


class Arccos(NpElemOperator):
    """Feature Arccos

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with arccos
    """

    def __init__(self, feature):
        super(Arccos, self).__init__(feature, "arccos")


class Arctan(NpElemOperator):
    """Feature Arctan

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with arctan
    """

    def __init__(self, feature):
        super(Arctan, self).__init__(feature, "arctan")


class Reciprocal(NpElemOperator):
    """Feature Reciprocal

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with reciprocal
    """

    def __init__(self, feature):
        super(Reciprocal, self).__init__(feature, "reciprocal")


class Clip(ElemOperator):
    """Feature Clipping (np.clip)

    Parameters
    ----------
    feature : Expression
        feature instance to be clipped
    a_min : float or array-like or None
        Minimum value. If None, only apply upper clipping via a_max.
    a_max : float or array-like or None
        Maximum value. If None, only apply lower clipping via a_min.

    Returns
    ----------
    Expression
        a feature instance with values clipped to [a_min, a_max]
    """

    def __init__(self, feature, a_min=None, a_max=None):
        if a_min is None and a_max is None:
            raise ValueError("At least one of a_min or a_max must be provided.")
        self.feature = feature
        self.a_min = a_min
        self.a_max = a_max

    def __str__(self):
        return "{}({}, a_min={}, a_max={})".format(
            type(self).__name__, self.feature, self.a_min, self.a_max
        )

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # Ensure numeric dtype (mirrors Sign's safeguard)
        series = series.astype(np.float32)

        # Support one-sided clipping if one bound is None (robust to older numpy)
        if self.a_min is None:
            return np.minimum(series, self.a_max)
        if self.a_max is None:
            return np.maximum(series, self.a_min)
        return np.clip(series, self.a_min, self.a_max)


Ext_OpsList = [Sqrt, Exp, Square, Sin, Cos, Tan, Tanh, Reciprocal, Clip]


for op in Ext_OpsList:
    if op not in OpsList:
        OpsList.append(op)
        print(f"Inject extended operator {op.__name__} into OpsList")
