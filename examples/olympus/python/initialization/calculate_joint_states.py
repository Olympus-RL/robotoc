from typing import Mapping,Tuple
from numpy.typing import NDArray

import numpy as np
np.set_printoptions(precision=3)


def calcualate_joint_states(robot, contact_status,base_pose: NDArray, theta_guess = None) -> NDArray:
    NUM_ITER = 100
    tol = 1e-8
    damp = 0
    alpha = 1.0
    max_dt = 10*np.pi/180
    dimf_ckc = robot.dimf_ckc()
    dimf_contact = contact_status.dimf()
    dimf = dimf_ckc + dimf_contact
    n_joints = robot.dimv() -6
    if theta_guess is  None:
        theta = np.zeros(n_joints)
    else:
        theta = theta_guess
    if dimf == 0:
        return theta
    q = np.concatenate((base_pose,theta))

    J_j = np.zeros((dimf,n_joints))
    C = np.zeros(dimf)

    for i in range(NUM_ITER):
        robot.update_kinematics(q)
        if dimf_ckc > 0:
            C[:dimf_ckc] = robot.get_ckc_residual()
            Jckc = robot.get_ckc_jacobian()
            J_j[:dimf_ckc,:] = Jckc[:,6:]
        if dimf_contact > 0:
            C[dimf_ckc:] = robot.get_contact_position_residual(contact_status)
            Jcontact = robot.get_contact_position_jacobian(contact_status)
            J_j[dimf_ckc:,:] = Jcontact[:,6:]
    
        if np.linalg.norm(C) < tol:
            break
     
        dt = - alpha*J_j.T.dot(np.linalg.solve(J_j.dot(J_j.T) + damp * np.eye(dimf), C))
        dt = np.clip(dt,-max_dt,max_dt)
        q[7:] += dt

    if i == NUM_ITER-1:
        print("Did not converge")
    
    return q[7:]






    






