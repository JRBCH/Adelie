# June 2021
# Author: J. Rossbroich

"""
misc.py

Contains functions used by the adelie package
"""


def getattr_deep(obj, attrs):
    """
    Like getattr(), but checks in children objects
    Example:
        X = object()
        X.Y = childobject()
        X.Y.variable = 5

        getattr(X, 'Y.variable') -> AttributeError
        getattr_deep(X, 'Y.variable') -> 5

    :param obj: object
    :param attrs: attribute (string)
    :return: attribute
    """
    for attr in attrs.split("."):
        obj = getattr(obj, attr)
    return obj

def hasattr_deep(obj, attrs):
    """
    Like hasattr(), but checks in children objects
    Example:
        X = object()
        X.Y = childobject()
        X.Y.variable = 5

        hasattr(X, 'Y.variable') -> False
        hasattr_deep(X, 'Y.variable') -> True

    :param obj: object
    :param attrs: attribute (string)
    :return: bool
    """
    try:
        getattr_deep(obj, attrs)
        return True
    except AttributeError:
        return False