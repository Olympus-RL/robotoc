from typing import Tuple, List
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation as R

from .opt_jump import OptJump as COptJump


class OptJumpTowr:
    """
    Python wrapper around the C++ OptJump class.
    Defined in opt_jump....so file.
    """

    _leg2idx = {"FL": 0, "FR": 1, "HL": 2, "HR": 3}  # need to check this
    _idx2ee = list(_leg2idx.keys())

    def __init__(self) -> None:
        self._optjump = COptJump()
        self._take_off_duration = 1.0

    def solve(self) -> bool:
        """
        Solves the optimization problem.
        Returns:
            bool: True if the optimization problem was solved successfully.
        """
        return self._optjump.solve()

    def set_initial_base_state(self, base_lin: NDArray, base_ang: NDArray) -> None:
        self._optjump.set_initial_base_state(base_lin, base_ang)

    def set_initial_EE_state(self, EE_pos: List[NDArray]) -> None:
        self._optjump.set_initial_EE_state(EE_pos)

    def set_takeoff_duration(self, takeoff_duration: float) -> None:
        self._take_off_duration = takeoff_duration
        self._optjump.set_takeoff_duration(takeoff_duration)

    def set_jump_length(self, jump_length: float) -> None:
        self._optjump.set_jump_length(jump_length)

    def get_base_state(self, t: float) -> Tuple[NDArray,NDArray]:
        """
        Retunes the optimized base state at time t.
        Args:
            t (float): time
        Returns:
            Tuple of NDArray: 
        """
        x_tower = self._optjump.get_base_state(t)
        euler = x_tower[3:6]
        base_rot = R.from_euler('zyx', euler)
        quat = base_rot.as_quat()
        q = np.concatenate((x_tower[:3], quat))
        euler_rate = x_tower[9:]
        R_z = R.from_euler('z', euler[2])
        R_zy = R.from_euler('zy', [euler[2], euler[1]])

        omega_x = np.array([1,0,0])*euler_rate[0]
        omega_y = np.array([0,1,0])*euler_rate[1]
        omega_z = np.array([0,0,1])*euler_rate[2]
        omega_w = omega_z + R_z.apply(omega_y) + R_zy.apply(omega_x)
    

        v_w = x_tower[6:9]
        v_b = base_rot.apply(v_w, inverse=True)
        omega_b = base_rot.apply(omega_w, inverse=True)

        omega_b = base_rot.apply(omega_w)
        v = np.concatenate((v_b, omega_b))

        return q, v
    
    def get_flight_duration(self) -> float:
        return self._optjump.get_flight_duration()

    

    def get_EE_pos(self, t: float) -> List[NDArray]:
        """
        Retunes the optimized end effector positions at time t.
        Args:
            t (float): time
        Returns:
            List of NDArray of size (3,) ie [x,y,z] pos of each ende effector.
        """
        return self._optjump.get_EE_pos(t)

    @staticmethod
    def ee2idx(ee: str) -> int:
        """
        Returns the index of the end effector.
        Args:
            ee (str): end effector name, ie "FL","FR","HL","HR"
        Returns:
            int: index of the end effector
        """
        return OptJumpTowr._leg2idx[ee]

    @staticmethod
    def idx2ee(idx: int) -> str:
        """
        Returns the name of the end effector.
        Args:
            idx (int): index of the end effector
        Returns:
            str: end effector name, ie "FL","FR","HL","HR"
        """
        return OptJumpTowr._idx2ee[idx]
