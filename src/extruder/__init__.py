"""Helper code for the extruder cut-frame pipeline (CIS 831, K-State).

Modules, in the order the notebook uses them:
    video     reading the camera videos (via ffmpeg)
    cuts      the 'blink' signal at hole A, cut rhythm, cut events, exact hole centre
    calib     die-plate map in millimetres, bolt refinement, pixel -> mm mapping
    prompt    the automatic box prompt for SAM 2
    track     SAM 2 tracking of one piece per cut, green / red / blue decision
    measure   size of a piece from its mask (mm or px)
    pipeline  the steps glued together, saving results
    viz       all figures
    ui        the click tool used in the notebook
"""
