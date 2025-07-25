import inspect
import numpy
from typing import Callable


def check_binary_matrices(*param_names: str) -> Callable:
    """
    Decorator to check that specified parameters are binary indicator matrices.

    Parameters
    ----------
    param_names : str
        Names of the parameters to look for.

    Returns
    -------
    decorator : Callable
        Function that wraps the function requiring the check.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get function signature and bind arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert specified arguments to numpy arrays
            arrays = [
                numpy.asarray(bound_args.arguments[param])
                for param in param_names
                if param in bound_args.arguments
            ]

            # Check row consistency
            for param, arr in zip(param_names, arrays):
                if arr.ndim != 2:
                    raise ValueError(f"Parameter '{param}' must be a 2D matrix.")
                if not numpy.all(numpy.isin(arr, [0, 1])):
                    raise ValueError(
                        f"Parameter '{param}' must be a binary indicator matrix (containing only 0s and 1s)."
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_same_columns(*param_names: str) -> Callable:
    """
    Decorator to check that specified parameters have the same number of columns.

    Parameters
    ----------
    param_names : str
        Names of the parameters to look for.

    Returns
    -------
    func : Callable
        Function that wraps the function requiring the check.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorator to check that specified parameters have the same number of columns.

        Parameters
        ----------
        func : Callable
            Function to wrap.

        Returns
        -------
        func : Callable
            Function that wraps the function requiring the check.
        """

        def wrapper(*args, **kwargs):
            """
            Wrapper function that checks the specified parameters for column consistency.

            Returns
            -------
            func : Callable
                Function that wraps the function requiring the check.
            """
            # Get function signature and bind arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert specified arguments to numpy arrays
            arrays = [
                numpy.asarray(bound_args.arguments[param])
                for param in param_names
                if param in bound_args.arguments
            ]

            # Check column consistency
            if len(arrays) > 1 and any(
                arr.shape[1] != arrays[0].shape[1] for arr in arrays
            ):
                raise ValueError(
                    f"All specified parameters ({', '.join(param_names)}) must have the same number of columns."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_same_rows(*param_names: str) -> Callable:
    """
    Decorator to check that specified parameters have the same number of rows.

    Parameters
    ----------
    param_names : str
        Names of the parameters to look for.

    Returns
    -------
    decorator : Callable
        Function that wraps the function requiring the check.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get function signature and bind arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert specified arguments to numpy arrays
            arrays = [
                numpy.asarray(bound_args.arguments[param])
                for param in param_names
                if param in bound_args.arguments
            ]

            # Check row consistency
            if len(arrays) > 1 and any(
                arr.shape[0] != arrays[0].shape[0] for arr in arrays
            ):
                raise ValueError(
                    f"All specified parameters ({', '.join(param_names)}) must have the same number of rows."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
