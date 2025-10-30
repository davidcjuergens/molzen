import numpy as np 

def parse_xyz(
    filename: str, xyz_th: bool = False, comment_parser: Callable = lambda x: x
) -> dict:
    """Parse an xyz file. NOTE: Does not work for QM9 .xyz files.
    Must basic format below.

    XYZ file format:

        -----------------
        <number of atoms>
        <comment line>
        <atom1> <x> <y> <z>
        <atom2> <x> <y> <z>
        ...
        -----------------

    Args:
        filename (str): The name of the file to parse
        xyz_th (bool): Return xyz coordinates as a torch tensor (else, numpy array)
        comment_parser (Callable): Function to parse comment line
    Returns:
        dict: Dictionary containing
    """

    elements, xyz = [], []

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                natoms = int(line)  # number of atoms
            elif i == 1:
                comment = line  # comment line
            else:
                element, *coords = line.strip().split()
                elements.append(element)
                xyz.append([float(coord) for coord in coords])

    # sanity checks
    assert len(elements) == natoms, (
        f".xyz file says {natoms} atoms, but found {len(elements)} atoms"
    )

    xyz = np.array(xyz) if not xyz_th else torch.tensor(xyz)

    if comment_parser is not None:
        comment = comment_parser(comment)

    return {"natoms": natoms, "comment": comment, "elements": elements, "xyz": xyz}
