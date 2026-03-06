"Orbital visualizations"

import py3Dmol


def plot_nto_cube(
    cube_filepath,
    isovalue=0.02,
    color_positive="cyan",
    color_negative="red",
    opacity=0.9,
    height=500,
    width=500,
):
    with open(cube_filepath, "r") as f:
        cube_data = f.read()

    #### py3dmol visualization code ####
    view = py3Dmol.view(width=width, height=height)
    view.addModel(cube_data, "cube")
    view.setStyle({"stick": {}})

    view.addVolumetricData(
        cube_data,
        "cube",
        {"isoval": isovalue, "color": color_positive, "opacity": opacity},
    )

    view.addVolumetricData(
        cube_data,
        "cube",
        {"isoval": -isovalue, "color": color_negative, "opacity": opacity},
    )

    view.zoomTo()
    view.show()

    return view
