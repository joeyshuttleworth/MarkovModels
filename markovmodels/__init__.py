import importlib
import os
import pkgutil
import inspect

__all__ = []

def import_and_expose_classes(package_name, package_path):
    for _, module_name, _ in pkgutil.walk_packages([package_path], prefix=package_name + "."):
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Only export classes defined in this module (not imported ones)
                if obj.__module__ == module_name:
                    globals()[name] = obj
                    __all__.append(name)
        except Exception as e:
            print(f"Failed to import from {module_name}: {e}")

import_and_expose_classes(__name__, os.path.dirname(__file__))
