# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:42:45 2023

@author: sb5
"""

from djitellopy import Tello

# Connect to the drone
drone = Tello()
drone.connect()

# Connect to your home WiFi
# drone.connect_to_wifi("MScA_Robotics", "drone100")
drone.connect_to_wifi("monikaBox", "0322March")

# reboot the drone
drone.reboot()