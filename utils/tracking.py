
import math
import time
from typing import Dict, List, Tuple

from numpy import double

from localization import Localization


class TrackedRobot:
    def __init__(self, initial_pos: Tuple[float,float]) -> None:
        self.positions: Dict[float, Tuple[float,float]] = {time.time():initial_pos} #time in miliseconds mapped to a Position tuple, TODO: make a Position object? or another class PositionHistory?


    def get_latest_position(self) -> Tuple[float,float]:
        return list(self.positions.items())[-1] # Dictionaries maintain insertion order, so we can do this to get most recent location

    def add_new_position(self, position: Tuple[int,int]):
        self.positions[time.time()] = position #TODO: Get time attached to frame data so we get the exact time that this was taken

    # TODO: add more methods for getting position in a useful format,ex: getting position data within a certain time range, etc.

class Tracker:
    def __init__(self, localization: Localization) -> None:
        self.robot_localizer = localization  #Localization object that supplies Tracker with our own robot's position
        self.tracked_robots: List[TrackedRobot] = []

    # Returns the final coordinates to move the turret given the TrackedRobot target
    def get_coords_to_move(self, target: TrackedRobot) -> None: #Change name
        # Use all given TrackedRobot information to calculate final aiming coordinates
        # Ex: Use location history to calculate velocity from the past 1 second --> offset based on that velocity
        pass


    def get_closest_tracked_robot(self) -> TrackedRobot:
        # return closest tracked robot just by their latest location
        pass

    # Updates state of Tracker with new plate locations so that it can figure out new locations for robots that its tracking
    def updateTrackerState(self, coords_depths_pairs):
        
        currentPosition = self.robot_localizer.get_robot_position()
        # Go through coords_depths_pairs and do math to find their location, as a tuple of coords
        plate_absolute_locations = []
        
        for pair in coords_depths_pairs:
            #[INSERT MATH HERE] (find angle, x = currentPosition[0] + sintheta*dist, y = currentPosition[1] + costheta*dist)
            pass
    
        # Now you have a bunch of absolute locations of plates and have to link them to previously tracked robots

        # TODO: need to think of logic needed of when we want to track, forget, etc. robots. That logic will most likely need to go here when the plates are being linked back to their original robots
        for tracked in self.tracked_robots:
            #find the closest plate location to tracked robot
            latest_position = tracked.get_latest_position()
            min = -1
            min_plate = None
            for plate_location in plate_absolute_locations:  #TODO improve this min finding
                min = min(min, math.dist(plate_location,latest_position))
                min_plate = plate_location
            
            if min != -1:
                tracked.add_new_position(min_plate)
                plate_absolute_locations.remove(min_plate)

        # Create new tracked robots from unused plate locations
        for plate_location in plate_absolute_locations:
            self.tracked_robots.append(TrackedRobot(plate_location))
            
        
