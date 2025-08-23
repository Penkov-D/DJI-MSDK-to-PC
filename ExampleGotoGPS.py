from OpenDJI import OpenDJI
from OpenDJI import EventListener

import keyboard
import cv2
import numpy as np
import re
import time

"""
In this example you will get a live video feedback, and be able to move the
drone between pre-defined GPS positions.

    press Q - to close the problem

    press W - to take off
    press S - to land

    press 1 - to move to the first GPS coordinate
    press 2 - to move to the second GPS coordinate
    press 3 - to move to the third GPS coordinate
"""


# GPS markers for the drone
# TODO : set desired GPS locations
POS_GPS_1 = (32.13291, 34.80572, )
POS_GPS_2 = (32.13306, 34.80836, )
POS_GPS_3 = (32.13568, 34.80792, )

# GPS position threshold - how close to the coordinate the drone
# have to be to register the point.
GPS_threshold = 3.0     # meters

# GPS timestamp threshold - how long can the gps be valid -
# in case the gps is lost.
GPS_timestamp_expired = 3.0     # seconds

# Movement factors
MOVE_VALUE = 0.025
ROTATE_VALUE = 0.15


# IP address of the connected android device
# TODO : set IP address according to the application
IP_ADDR = "192.168.1.184"

# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 0.25

# Set to true to show live GPS and compass data
DEBUG_OUTPUT = True


################################ FRAME LISTENER ################################

# Initiate frame as blank frame
frame = np.zeros((1080, 1920, 3))
frame = cv2.putText(frame, "No Image", (200, 300),
                    cv2.FONT_HERSHEY_PLAIN, 10,
                    (255, 255, 255), 10)
frame = cv2.resize(frame, dsize = None,
                   fx = SCALE_FACTOR, fy = SCALE_FACTOR)

# Create background listener for video
class frameListener(EventListener):

    def onValue(self, _frame):
        """ Called when new frame available """
        global frame
        frame = cv2.resize(_frame, dsize = None,
                           fx = SCALE_FACTOR, fy = SCALE_FACTOR)

    def onError(self, ):
        pass



################################# GPS LISTENER #################################

p_isSet     = False
p_latitude  = 0.0   # degree
p_longitude = 0.0   # degree
p_altitude  = 0.0   # meters
p_timestamp = 0.0   # seconds

# Regex to extract decimal numbers
NUM_REG = '[-+]?\\d+\\.?\\d*'

# Create background listener for GPS position
class gpsListener(EventListener):

    def onValue(self, _location3D):
        """ Called when new GPS coordinate is available """
        
        # # Example to see if the regex compiles
        # location3D = '{"latitude":32.1125,"longitude":34.805,"altitude":20}'
        global p_isSet, p_latitude, p_longitude, p_altitude, p_timestamp

        # Phrase the returned string
        location_pattern = re.compile(
            '{"latitude":(' + NUM_REG + '),' +
            '"longitude":(' + NUM_REG + '),' +
            '"altitude":(' + NUM_REG + ')}')
        
        # If the result match the regex, parse it.
        location_match : re.Match = location_pattern.fullmatch(_location3D)
        if location_match is not None:

            # Extract the location parameters
            p_isSet     = True
            p_latitude  = float(location_match.group(1))
            p_longitude = float(location_match.group(2))
            p_altitude  = float(location_match.group(3))
            p_timestamp = time.time()

            # Print location parameters:
            if DEBUG_OUTPUT:
                print(f"Latitude : {p_latitude:.6f}, " +
                      f"longitude : {p_longitude:.6f}, " +
                      f"altitude : {p_altitude:.6f}")

        # Else, there might have been an error.
        else:
            if DEBUG_OUTPUT:
                print("Error while receiving GPS coordinates.")

    def onError(self, ):
        # TODO : change parameters of onError
        pass



############################### COMPASS LISTENER ###############################

c_isSet     = False
c_bearing   = 0.0   # degree
c_timestamp = 0.0   # seconds

# Create background listener for compass
class compassListener(EventListener):

    def onValue(self, _compass):
        """ Called when new compass measurement is available """
        
        # # Example of return message
        # compass = '-12.4'
        global c_isSet, c_bearing, c_timestamp

        try:
            # Extract compass measurement
            c_bearing = float(_compass)
            c_timestamp = time.time()
            c_isSet = True

            if DEBUG_OUTPUT:
                print(f"Bearing: {c_bearing:.2f}")

        except:
            if DEBUG_OUTPUT:
                print("Error while receiving compass measurements.")

    def onError(self, ):
        # TODO : change parameters of onError
        pass



################################# GPS NAVIGATOR ################################

# delay between moving the drone again
COMMANDS_DELAY = 0.5    # seconds

# Earth's radius
EARTH_RADIUS = 6371e3   # meters

# Calculate bearing between two gps coordinates
# https://www.movable-type.co.uk/scripts/latlong.html
def calc_bearing (latitude_1, longitude_1, latitude_2, longitude_2):

    # Convert all parameters to radians
    latitude_1  = np.deg2rad(latitude_1)
    longitude_1 = np.deg2rad(longitude_1)
    latitude_2  = np.deg2rad(latitude_2)
    longitude_2 = np.deg2rad(longitude_2)

    # arc tangent parameters
    y = np.sin(longitude_2 - longitude_1) * np.cos(latitude_2)
    x = np.cos(latitude_1) * np.sin(latitude_2) -   \
        np.sin(latitude_1) * np.cos(latitude_2) * np.cos(longitude_2 - longitude_1)
    
    # Calculate bearing
    theta = np.atan2(y, x)
    theta = np.rad2deg(theta)

    return theta


# Calculate distance between two gps coordinates
# https://www.movable-type.co.uk/scripts/latlong.html
def calc_distance (latitude_1, longitude_1, latitude_2, longitude_2):

    # Convert all parameters to radians
    latitude_1  = np.deg2rad(latitude_1)
    longitude_1 = np.deg2rad(longitude_1)
    latitude_2  = np.deg2rad(latitude_2)
    longitude_2 = np.deg2rad(longitude_2)

    delta_latitude  = latitude_2  - latitude_1
    delta_longitude = longitude_2 - longitude_1

    a = np.sin(delta_latitude / 2) * np.sin(delta_latitude / 2) +   \
        np.cos(latitude_1) * np.cos(latitude_2) *   \
        np.sin(delta_longitude / 2) ** 2
    
    distance = EARTH_RADIUS * 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))
    return distance


# Move to a designated GPS coordination
def gotoGPS (drone : OpenDJI, latitude : float, longitude : float):
    """
    Move the drone to a desired GPS location.

    Args:
        drone (OpenDJI): the drone object to control.
        latitude (float): the latitude destination.
        longitude (float): the longitude destination.
    
    Return:
        result (bool): if succeeded or not
    """

    # Check if compass and gps available
    if not c_isSet: return False
    if not p_isSet: return False

    # Enable application control
    drone.enableControl()

    while calc_distance(p_latitude, p_longitude, latitude, longitude) > GPS_threshold:
        
        # Check if GPS didn't got updated
        if time.time() - p_timestamp > GPS_timestamp_expired:
            drone.move(0, 0, 0, 0)
            drone.disableControl()
            return False
        
        # Calculate relative bearing
        d_bearing = calc_bearing(p_latitude, p_longitude, latitude, longitude)
        r_bearing = d_bearing - c_bearing

        # Calculate relative force
        fb_force = np.cos(np.deg2rad(r_bearing)) * MOVE_VALUE
        lr_force = np.sin(np.deg2rad(r_bearing)) * MOVE_VALUE

        # Move the drone
        drone.move(0, 0, lr_force, fb_force)

        if DEBUG_OUTPUT:
            print(f"MOVE: {fb_force:.3f}, {lr_force:.3f}")
        
        # Small delay to not spam movements
        time.sleep(COMMANDS_DELAY)

    # End drone movement
    drone.move(0, 0, 0, 0)
    drone.disableControl()
    return True



##################################### MAIN #####################################

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    
    # Register background listeners for video, GPS and compass
    drone.frameListener(frameListener())
    drone.listen(OpenDJI.MODULE_FLIGHTCONTROLLER, "AircraftLocation3D", gpsListener())
    drone.listen(OpenDJI.MODULE_FLIGHTCONTROLLER, "CompassHeading", compassListener())
    
    # Press 'q' to close the program
    print("Press 'q' to close the program")
    while not keyboard.is_pressed('q'):

        # Show frame
        cv2.imshow("Live video", frame)
        cv2.waitKey(20)     # Set 50 fps
        # TODO : manipulate the frames

        # Takeoff and land commands
        if keyboard.is_pressed('w'): print("Takeoff:", drone.takeoff(True))
        if keyboard.is_pressed('s'): print("Land:",    drone.land(True))

        # Goto GPS examples
        if keyboard.is_pressed('1'): gotoGPS(drone, *POS_GPS_1)
        if keyboard.is_pressed('2'): gotoGPS(drone, *POS_GPS_2)
        if keyboard.is_pressed('3'): gotoGPS(drone, *POS_GPS_3)