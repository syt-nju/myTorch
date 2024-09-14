'''
Create by SkywardSigil
Date: 2024-09-14 16:00:00
Last Modified: 2024-09-14 16:10:00
Description:
This file is used to set the root path of the project
and add it to the system path.
This is a temporary solution, and it will be replaced by a more elegant solution later.
'''


import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0] = root_path