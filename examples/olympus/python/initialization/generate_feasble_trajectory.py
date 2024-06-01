from typing import List,Tuple,Mapping
from numpy.typing import NDArray

from math import sqrt
import numpy as np
from scipy.optimize import linprog

import robotoc


def make_sol_feaseble(robot,configuration,td,contact_sequence) -> List[robotoc.SplitSolution]:
    assert(len(configuration) == len(td))
    sol = [robotoc.SplitSolution(robot) for _ in range(len(td))]

    for i in range(len(td)-1,-1,-1):
        grid = td[i]
        dt = grid.dt


        if grid.type == robotoc.GridType.Terminal:
            q = configuration[i]
            v = np.zeros(robot.dimv())
            sol[i].q = q
            sol[i].v = v

        elif (grid.type == robotoc.GridType.Intermediate
            or grid.type == robotoc.GridType.Lift):

            q = configuration[i]   
            v = robot.subtract_configuration(sol[i+1].q,q)/dt
            a = (sol[i+1].v - v)/dt
            u,F_ckc,F_contact = ID(robot,q,v,a,contact_sequence.contact_status(grid.phase),robot.joint_effort_limit())
            sol[i].q = q
            sol[i].v = v
            sol[i].a = a
            sol[i].u = u
            sol[i].f_ckc = F_ckc
            sol[i].f_contact = F_contact
            sol[i].set_contact_status(contact_sequence.contact_status(grid.phase))
            sol[i].set_f_stack()

        elif grid.type == robotoc.GridType.Impact:
            q = sol[i+1].q
            dv = np.zeros(robot.dimv())
            v = sol[i+1].v - dv
            sol[i].q = q
            sol[i].v = v
            sol[i].dv = dv 
            sol[i].set_contact_status(contact_sequence.impact_status(grid.impact_index))
            sol[i].set_f_stack()
      
        
        if grid.switching_constraint:
            grid_next_next = td[i+2]
            sol[i].set_switching_constraint_dimension(contact_sequence.impact_status(grid_next_next.impact_index).dimf())
        else:
            sol[i].set_switching_constraint_dimension(0)
    return sol            
                
def ID(robot,q,v,a,contact_status,torque_limits) -> Tuple[NDArray,List[NDArray],List[NDArray]]:
    robot.update_kinematics(q,v,a)

    num_contacts = 0
    mus = contact_status.friction_coefficients()
    for i in range(contact_status.max_num_contacts()):
        if contact_status.is_contact_active(i):
            num_contacts += 1

    dim_f_contact = contact_status.dimf()
    dim_f_ckc = robot.dimf_ckc()
    dim_f = dim_f_contact + dim_f_ckc
    dim_u = robot.dimu()

    if dim_f == 0:
        return robot.rnea(q,v,a)[6:],[np.zeros(2) for _ in range(robot.numCKCs())],[np.zeros(6) for _ in range(robot.max_num_contacts())]

    b_friction_cone = np.zeros(5*num_contacts)
    A_friction_cone = np.zeros((5*num_contacts,3*num_contacts))
    c_slack_friction_cone = np.ones(5*num_contacts)
    
    bounds_friction_cone = []
    f_dim = 0
    cone_dim = 0

    for i in range(num_contacts): 
        mu = mus[i]
        A_friction_cone[cone_dim:cone_dim+5,f_dim:f_dim+3] = np.array([
            [-1,0,-mu/sqrt(2)], # -mu*f_z <= f_x
            [1,0,-mu/sqrt(2)],  # f_x <= mu*f_z
            [0,-1,-mu/sqrt(2)], # -mu*f_z <= f_y
            [0,1,-mu/sqrt(2)],  # f_y <= mu*f_z
            [0,0,-1]])  # 0 <= f_z
        bounds_friction_cone.append((None,None))
        bounds_friction_cone.append((None,None))
        bounds_friction_cone.append((None,None))
        c_slack_friction_cone[cone_dim:cone_dim+4] = 1e4
        c_slack_friction_cone[cone_dim+4] = 1e4
        f_dim += 3
        cone_dim += 5

    bounds_u = []
    A_u = np.zeros((2*dim_u,dim_u))
    b_u = np.zeros(2*dim_u)
    c_slack_u = np.ones(2*dim_u)*0
    for i in range(dim_u):
        bounds_u.append((None,None))
        A_u[2*i:2*i+2,i]=np.array([1,-1]) #u_i <= torque_limits[i] and -u_i <= torque_limits[i] 
        b_u[2*i:2*i+2] = np.array([torque_limits[i],torque_limits[i]])

    J_contact = robot.get_contact_position_jacobian(contact_status)
    J_ckc = robot.get_ckc_jacobian()
    u_virtual = robot.rnea(q,v,a)

    bounds_ckc = [(None,None) for _ in range(dim_f_ckc)]
    c = np.zeros(dim_u+dim_f)


    num_ineq = 2*dim_u + 5*num_contacts
    dim_x = dim_u + dim_f
    A_ineq = np.zeros((num_ineq,dim_x))
    b_ineq = np.zeros(num_ineq)
    A_ineq[:2*dim_u,:dim_u] = A_u
    b_ineq[:2*dim_u] = b_u
    A_ineq[2*dim_u:,dim_u+dim_f_ckc:] = A_friction_cone
    b_ineq[2*dim_u:] = b_friction_cone

    num_eq = robot.dimv()
    A_eq = np.zeros((num_eq,dim_x))
    b_eq = u_virtual
    A_eq[:,:dim_u] = robot.S().T
    A_eq[:,dim_u:dim_u+dim_f_ckc] = J_ckc.T
    A_eq[:,dim_u+dim_f_ckc:] = J_contact.T

    I_slack = np.eye(num_ineq)
    c_slack = np.concatenate([c_slack_u,c_slack_friction_cone])
    bounds_slack = [(0,None) for _ in range(num_ineq)]


    c = np.concatenate((c,c_slack))
    bounds =bounds_u + bounds_ckc + bounds_friction_cone + bounds_slack

    A_ineq = np.concatenate((A_ineq,-I_slack),axis=1)
    A_eq = np.concatenate((A_eq,np.zeros((num_eq,num_ineq))),axis=1)


    res = linprog(c=c,A_ub=A_ineq,b_ub=b_ineq,bounds=bounds,A_eq=A_eq,b_eq=b_eq)
    u = res.x[:dim_u]
    f_ckc = res.x[dim_u:dim_u+dim_f_ckc]
    f_contact_world = res.x[dim_u+dim_f_ckc:]
    F_contact = []
    F_ckc = []

    f_dim = 0
    for i in range(num_contacts):
        if contact_status.is_contact_active(i):
            frame_name = contact_status.contact_frame_name(i)
            R = robot.frame_rotation(frame_name)
            f_c = f_contact_world[f_dim:f_dim+3]
            if f_c[2] < 0:
                f_c[2] = 0
            f_i = np.zeros(6)
            f_i[:3] = R.T @ f_c
            F_contact.append(f_i)
            f_dim += 3 
        else:
            F_contact.append(np.zeros(6))
    for i in range(int(dim_f_ckc/2)):
        F_ckc.append(f_ckc[2*i:2*i+2].copy())

    ## residuals

    res_id = np.linalg.norm(A_eq[:,:-num_ineq] @ res.x[:-num_ineq] - b_eq)
    res_ineq = (A_ineq[:,:-num_ineq] @ res.x[:-num_ineq] - b_ineq).clip(min=0)
    res_friction_cone = np.linalg.norm(res_ineq[-5*num_contacts:])
    res_u = np.linalg.norm(res_ineq[:2*dim_u])

    if res_id + res_friction_cone + res_u > 1e-3 and False:
        print("Warning: residuals are large")
        print("residuals ID: ", res_id)
        print("residuals friction cone:", res_friction_cone)
        print("residuals u:", res_u)
        print("="*20)
    
    return u,F_ckc,F_contact







      
    
        



            

    





