from OpenDJI import OpenDJI

import re

"""
GPS coordinate example.
"""

# IP address of the connected android device
IP_ADDR = "192.168.1.184"

# Regex to extract decimal numbers
NUM_REG = '[-+]?\\d+\\.?\\d*'

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    
    # Get the location info
    location3D = drone.getValue(OpenDJI.MODULE_FLIGHTCONTROLLER, "AircraftLocation3D")
    print("Original result :", location3D)

    # # Example to see if the regex compiles
    # location3D = '{"latitude":32.1125,"longitude":34.805,"altitude":20}'

    # you need to manually check for errors, and phrase the returned string
    location_pattern = re.compile(
        '{"latitude":(' + NUM_REG + '),' +
        '"longitude":(' + NUM_REG + '),' +
        '"altitude":(' + NUM_REG + ')}')

    # If the result match the regex, parse it.
    location_match : re.Match = location_pattern.fullmatch(location3D)
    if location_match is not None:

        # Extract the location parameters
        latitude  = float(location_match.group(1))
        longitude = float(location_match.group(2))
        altitude  = float(location_match.group(3))

        # Print location parameters:
        print("Latitude :",  latitude)
        print("longitude :", longitude)
        print("altitude :",  altitude)
    
    # Else, there might have been an error.
    else:
        print("Error while receiving GPS coordinates.")

    print()