from typing import List,Tuple,Mapping
from numpy.typing import NDArray

from math import sqrt
import numpy as np
from scipy.optimize import linprog

import robotoc


def generate_feaseble_trajectory(robot,configuration: List[NDArray],dts: List[float],contact_pos :Mapping[str,NDArray],mu:float) -> List[robotoc.SplitSolution]:
    base_vel = []
    base_acc = []
    sol = []
    for i, dt in enumerate(dts):
        v = robot.subtract_configuration(configuration[i+1], configuration[i]) / dt
        base_vel.append(v)
    base_vel.append(base_vel[-1].copy())

    for i, dt in enumerate(dts):
        a = (base_vel[i+1] - base_vel[i]) / dt
        base_acc.append(a)
    base_acc.append(base_acc[-1].copy())

    num_contacts = len(contact_pos)

    if num_contacts:
        J_c = np.zeros((3*num_contacts,robot.dimv()))
        for i,(q,v,a) in enumerate(zip(configuration,base_vel,base_acc)):
            robot.forward_kinematics(q,v,a,True)
            A_friction_cone = np.zeros((5*num_contacts,3*num_contacts))
            b_friction_cone = np.zeros(5*num_contacts)
            bounds_friction_cone = []
            f_dim = 0
            cone_dim = 0
            for frame_name, pos in contact_pos.items(): 
                J = robot.get_frame_world_jacobian(frame_name)
                J_c[f_dim:f_dim+3,:] = J[:3,:] 
                A_friction_cone[cone_dim:cone_dim+5,f_dim:f_dim+3] = np.array([
                    [-1,0,-mu/sqrt(2)], # -mu*f_z <= f_x
                    [1,0,-mu/sqrt(2)],  # f_x <= mu*f_z
                    [0,-1,-mu/sqrt(2)], # -mu*f_z <= f_y
                    [0,1,-mu/sqrt(2)],  # f_y <= mu*f_z
                    [0,0,-1]])  # 0 <= f_z
                bounds_friction_cone.append((None,None))
                bounds_friction_cone.append((None,None))
                bounds_friction_cone.append((None,None))
                f_dim += 3
                cone_dim += 5
            u_virtual = robot.rnea(q,v,a)
            u_virtual_base = u_virtual[:6]
            J_c_base = J_c[:,:6]


            c = np.zeros(3*num_contacts)

            #add slack on friction cone constraints
            num_slack =A_friction_cone.shape[0]
            I_slack = np.eye(num_slack)
            c_slack = np.ones(num_slack)*1e4
            bounds_slack = [(0,None)]*num_slack
            A_friction_cone = np.concatenate((A_friction_cone,-I_slack),axis=1)
            bounds_friction_cone += bounds_slack
            c = np.concatenate((c,c_slack))

            A_eq = np.concatenate((J_c_base.T,np.zeros((6,num_slack))),axis=1)

            res = linprog(c=c,A_ub=A_friction_cone,b_ub=b_friction_cone,bounds=bounds_friction_cone,A_eq=A_eq,b_eq=u_virtual_base)
            F_c_world = res.x[:3*num_contacts]
            F_b = []

            f_dim = 0
            for frame_name, pos in contact_pos.items(): 
                R = robot.frame_rotation(frame_name)
                f_c = F_c_world[f_dim:f_dim+3]
                f_i = np.zeros(6)
                f_i[:3] = R.T @ f_c
                F_b.append(f_i)

            u = u_virtual[6:] - (J_c[:,6:]).T @ F_c_world

            residual_ID = u_virtual - J_c.T @ F_c_world 
            residual_ID[6:] -= u

            residual_FC = (A_friction_cone @ res.x - b_friction_cone).clip(min=0)

            s_t = robotoc.SplitSolution(robot)
            s_t.q = q
            s_t.v = v
            s_t.u = u
            s_t.f = F_b
            s_t.set_f_stack()
            sol.append(s_t)
            
            s_t.a = a

            if not res.success:
                print("=====================================")
                print("Failed to solve for contact forces")
                print("F: ",  F_b)
                print("q: ", q)
                print("u_virtual: ", u_virtual)
                print("u: ", u)
                print("a: ", a)
                print("v: ", v)

    return sol
        



            

    





