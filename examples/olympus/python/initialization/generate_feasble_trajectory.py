from typing import List,Tuple,Mapping
from numpy.typing import NDArray

from math import sqrt
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import osqp

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
            v[6:] = np.clip(v[6:],-robot.joint_velocity_limit(),robot.joint_velocity_limit())
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

        assert(len(sol[i].f_contact) == robot.max_num_contacts())
        assert(len(sol[i].f_ckc) == robot.numCKCs())
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
        u = np.clip(robot.S() @ robot.rnea(q,v,a),-torque_limits+2,torque_limits-2)
        return u,[np.zeros(2) for _ in range(robot.numCKCs())],[np.zeros(6) for _ in range(robot.max_num_contacts())]

    b_friction_cone = np.zeros(5*num_contacts)
    A_friction_cone = np.zeros((5*num_contacts,3*num_contacts))
    c_slack_friction_cone = np.ones(5*num_contacts)
    
    l_friction_cone = np.zeros(5*num_contacts)
    u_friction_cone = np.ones(5*num_contacts)*1e4
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
        l_friction_cone[cone_dim:cone_dim+5] = -np.inf
        u_friction_cone[cone_dim:cone_dim+5] = 0
        f_dim += 3
        cone_dim += 5

    bounds_u = []
    A_u = np.eye(dim_u)
    l_u = -torque_limits
    u_u = torque_limits
    
    J_contact = robot.get_contact_position_jacobian(contact_status)
    J_ckc = robot.get_ckc_jacobian()
    u_virtual = robot.rnea(q,v,a)
    u_id = u_virtual
    l_id = u_virtual


    L = np.concatenate((l_id,l_u,l_friction_cone))
    U = np.concatenate((u_id,u_u,u_friction_cone))

    num_ineq = dim_u + 5*num_contacts
    dim_x = dim_u + dim_f
    A_ineq = np.zeros((num_ineq,dim_x))
    b_ineq = np.zeros(num_ineq)
    A_ineq[:dim_u,:dim_u] = A_u
    A_ineq[dim_u:,dim_u+dim_f_ckc:] = A_friction_cone
    

    num_eq = robot.dimv()
    A_eq = np.zeros((num_eq,dim_x))
    A_eq[:,:dim_u] = robot.S().T
    A_eq[:,dim_u:dim_u+dim_f_ckc] = J_ckc.T
    A_eq[:,dim_u+dim_f_ckc:] = J_contact.T


    #slack on eq 
    num_slack = num_eq
    I_slack = np.eye(num_slack)
    P_slack = I_slack*1e4

    A_slack = np.zeros((num_eq+num_ineq,dim_x+num_slack))
    A_slack[:num_eq,:dim_x] = A_eq
    A_slack[:num_eq,dim_x:] = I_slack
    A_slack[num_eq:,:dim_x] = A_ineq
    P = np.zeros((dim_x+num_slack,dim_x+num_slack))
    P[dim_x:,dim_x:] = P_slack

    P_sparse = sparse.csc_matrix(P)
    A_sparse = sparse.csc_matrix(A_slack)
    m = osqp.OSQP()
    m.setup(P=P_sparse, q=np.zeros(dim_x+num_slack), A=A_sparse, l=L, u=U,verbose=False,warm_start=True)
    result = m.solve()
    u = result.x[:dim_u]
    f_ckc = result.x[dim_u:dim_u+dim_f_ckc]
    f_contact_world = result.x[dim_u+dim_f_ckc:]
    F_contact = []
    F_ckc = []

    f_dim = 0
    for i in range(robot.max_num_contacts()):
        if contact_status.is_contact_active(i):
            frame_name = contact_status.contact_frame_name(i)
            R = robot.frame_rotation(frame_name)
            f_c = f_contact_world[f_dim:f_dim+3].copy()
            if f_c[2] < 0:
                f_c[:] = 0
            if np.abs(f_c[0]) > mus[i]/np.sqrt(2)*f_c[2]:
                f_c[0] = np.sign(f_c[0])*mus[i]/np.sqrt(2)*f_c[2]
            if np.abs(f_c[1]) > mus[i]/np.sqrt(2)*f_c[2]:
                f_c[1] = np.sign(f_c[1])*mus[i]/np.sqrt(2)*f_c[2]
            
            f_i = np.zeros(6)
            f_i[:3] = R.T @ f_c
            F_contact.append(f_i)
            f_dim += 3 
        else:
            F_contact.append(np.zeros(6))
    for i in range(int(dim_f_ckc/2)):
        F_ckc.append(f_ckc[2*i:2*i+2].copy())

    ## residuals

    #res_id = np.linalg.norm(A_eq[:,:-num_ineq] @ result.x[:-num_ineq] - b_eq)
    #res_ineq = (A_ineq[:,:-num_ineq] @ result.x[:-num_ineq] - b_ineq).clip(min=0)
    #res_friction_cone = np.linalg.norm(res_ineq[-5*num_contacts:])
    #res_u = np.linalg.norm(res_ineq[:2*dim_u])
#
    #if res_id + res_friction_cone + res_u > 1e-3 and False:
    #    print("Warning: residuals are large")
    #    print("residuals ID: ", res_id)
    #    print("residuals friction cone:", res_friction_cone)
    #    print("residuals u:", res_u)
    #    print("="*20)
    
    return np.clip(u,-torque_limits+2,torque_limits-2),F_ckc,F_contact







      
    
        



            

    





