"""General helper objects"""

from molzen.ptable import symbol_to_z


def get_atomic_number(element: str) -> int:
    """Get the atomic number for a given element symbol.

    Args:
        element (str): The element symbol (e.g., 'H', 'He', 'Li').
    """
    element = element.capitalize()
    return symbol_to_z[element]
