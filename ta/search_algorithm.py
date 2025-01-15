from importlib import import_module


def create_algorithm(ta_name, **kwargs):
    try:
        module = import_module("." + ta_name, "ta")
        ta_class = getattr(module, ta_name)
        ta = ta_class(**kwargs)
        return ta
    except ImportError:
        raise ImportError(f"Class '{ta_name}' not found or import failed.")
