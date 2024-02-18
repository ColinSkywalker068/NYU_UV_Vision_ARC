"""
Marker locations in the real orld coordinate system
Tx, Ty, Tz is the location of the marker's center from the origin
o, p, k stand for omega, phi and kappa which are rotation of the markers plane about the X, Y and Z axes respectively
"""
# rotations are clockwise in degrees

# World coordinate, bottom left of page 25: https://rm-static.djicdn.com/tem/13194/RoboMaster%202023%20University%20League%20Rules%20Manual%20V1.0%20(20221118).pdf


locs = {
    "R_A": {
        "Tx": 4575,
        "Ty": 4125,
        "Tz": 225,
        "o": 0,
        "p": 0,
        "k": 270,
    },
    "R_B": {
        "Tx": 1925,
        "Ty": 0,
        "Tz": 200,
        "o": 0,
        "p": 0,
        "k": 0,
    },
    "R_C": {
        "Tx": 0,
        "Ty": 1858,
        "Tz": 200,
        "o": 0,
        "p": 0,
        "k": 90,
    },
    "R_D": {
        "Tx": 425,
        "Ty": 8000,
        "Tz": 200,
        "o": 0,
        "p": 0,
        "k": 180,
    },
    "R_E": {
        "Tx": 7425,
        "Ty": 3475,
        "Tz": 225,
        "o": 0,
        "p": 0,
        "k": 270,
    },
    # blue is just flipped on y and x, and the directions are opposite
    # x=12000, y=8000
    # middle of field is (6000, 4000)
    # NEWX: (12000-x)
    # NEWY: (8000-y)
    # NEWK: (180-k)
    # Frame of reference is still the same though
    "BL_A": {
        "Tx": 7425,
        "Ty": 3875,
        "Tz": 225,
        "o": 0,
        "p": 0,
        "k": 90,
    },
    "BL_B": {
        "Tx": 10075,
        "Ty": 8000,
        "Tz": 225,
        "o": 0,
        "p": 0,
        "k": 180,
    },
    "BL_C": {
        "Tx": 12000,
        "Ty": 6142,
        "Tz": 200,
        "o": 0,
        "p": 0,
        "k": 270,
    },
    "BL_D": {
        "Tx": 11575,
        "Ty": 0,
        "Tz": 200,
        "o": 0,
        "p": 0,
        "k": 0,
    },
    "BL_E": {
        "Tx": 4575,
        "Ty": 4525,
        "Tz": 200,
        "o": 0,
        "p": 0,
        "k": 90,
    },
}
