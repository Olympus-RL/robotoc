from typing import Mapping,Tuple
from numpy.typing import NDArray

import numpy as np
np.set_printoptions(precision=3)


def calcualate_joint_states(robot, contact_pos :Mapping[str,NDArray],base_pose: NDArray, theta_guess = None) -> NDArray:
    #chekc_finite_diff(robot,contact_pos, np.concatenate((base_pose,theta_guess)))
    #test(robot,contact_pos, np.concatenate((base_pose,theta_guess)))
    NUM_ITER = 100
    tol = 1e-8
    damp = 0
    alpha = 1.0
    max_dt = 10*np.pi/180
    num_contacts = len(contact_pos)
    n_joints = robot.dimv() -6
    if theta_guess is  None:
        theta = np.zeros(n_joints)
    else:
        theta = theta_guess


    q = np.concatenate((base_pose,theta))

    J = np.zeros((6,robot.dimv()))
    J_b = np.zeros((3*num_contacts,6))
    J_j = np.zeros((3*num_contacts,n_joints))
    C = np.zeros(3*num_contacts)

    
    for i in range(NUM_ITER):
        robot.forward_kinematics(q,True)
        idx = 0
        for frame_name, pos in contact_pos.items(): 
            J[:] = robot.get_frame_world_jacobian(frame_name)
            J_j[idx:idx+3,:] = J[:3,6:]
            J_b[idx:idx+3,:] = J[:3,:6]
            C[idx:idx+3] = robot.frame_position(frame_name) - pos
            idx += 3
        if np.linalg.norm(C) < tol:
            break
     
        dt = - alpha*J_j.T.dot(np.linalg.solve(J_j.dot(J_j.T) + damp * np.eye(n_joints), C))
        dt = np.clip(dt,-max_dt,max_dt)
        q[7:] += dt

    if i == NUM_ITER-1:
        print("Did not converge")

    
    return q[7:]






    






