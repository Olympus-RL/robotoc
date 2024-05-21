import numpy as np
from robotoc.cost.task_space_6d_ref_base import TaskSpace6DRefBase
from robotoc.robot.se3 import SE3



class TrajectoryRef(TaskSpace6DRefBase):
    def __init__(self):
        TaskSpace6DRefBase.__init__(self)
        self._pose = SE3(np.array([0,0,0,1.0]), np.array([0,0,0]))

    def isActive(self, grid_info) -> bool:
        return True
    
    def updateRef(self,grid_info,ref):
        print("updateRef")
        ref.R = self._pose.R
        ref.trans = self._pose.trans 
        return self._pose
        

    
