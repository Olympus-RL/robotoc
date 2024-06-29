import robotoc
from tojr import OptJumpTowr
from initialization import calcualate_joint_states,make_sol_feaseble,ID
from logger import solution_to_blender_json
from robotoc.cost.periodic_com_ref import TrajectoryRef
import numpy as np
import math
import matplotlib.pyplot as plt
import tikzplotlib
import scipy.spatial.transform.rotation as R


def test_jump(jump_len):

    model_info = robotoc.RobotModelInfo()
    model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus.urdf"
    model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
    baumgarte_time_step_contact = 0.05
    baumgarte_time_step_ckc = 0.005

    model_info.point_contacts = [robotoc.ContactModelInfo('FrontLeft_paw', baumgarte_time_step_contact),
                                 robotoc.ContactModelInfo('BackLeft_paw', baumgarte_time_step_contact),
                                 robotoc.ContactModelInfo('FrontRight_paw', baumgarte_time_step_contact),
                                 robotoc.ContactModelInfo('BackRight_paw', baumgarte_time_step_contact)]
    model_info.ckcs = [robotoc.CKCInfo( "FrontLeft_ankle_outer","FrontLeft_ankle_inner",baumgarte_time_step_ckc),
                    robotoc.CKCInfo("FrontRight_ankle_outer","FrontRight_ankle_inner",baumgarte_time_step_ckc),
                    robotoc.CKCInfo("BackLeft_ankle_outer","BackLeft_ankle_inner",baumgarte_time_step_ckc),
                    robotoc.CKCInfo("BackRight_ankle_outer","BackRight_ankle_inner",baumgarte_time_step_ckc)]

    #model_info.ckcs = []
    model_info.contact_inv_damping = 1.0e-12
    robot = robotoc.Robot(model_info)
    lower_limits = np.pi/180*np.array([ -20, -20., -30., -120, -240,  #back left
                                        -20, -120, -30., -20., -240, #back right
                                        -20, -20., -240, -120,  -30, #front left
                                        -20, -20., -240, -20.,  -30]) #front right
                                              ###inner###  ###outer###
    upper_limits = np.pi/180*np.array([ 20, 120, 240, 20., 30.,  #back left
                                        20, 20., 240, 120, 30., #back right
                                        20, 120, 30., 20., 240, #front left
                                        20, 120, 30., 120, 240]) #front right
                                            ###inner###  ###outer###
    #lower_limits[:] = -4.0
    #upper_limits[:] = 4.0



    robot.set_lower_joint_position_limit(lower_limits)
    robot.set_upper_joint_position_limit(upper_limits)
    robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
    joint_efforts_limit = np.full(robot.dimu(), 24.0)
    robot.set_joint_effort_limit(joint_efforts_limit)
    robot.set_gravity(-3.72)
    robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
    robot.set_joint_effort_limit(joint_efforts_limit)

    dt = 0.005
    jump_length = np.array([jump_len, 0, 0])
    take_off_duration = 1.0
    flight_duration = 0.20
    touch_down_duration = 0.5
    t0 = 0.
    use_sto = False

    q_standing = np.array([0., 0., 0.30, 0.0, 0., 0., 1.0, 
                             0.0,  1.4,  2., -1., -1.6,  #back left
                            -0.0, -1.4,  2.,  1., -1.6, #back right
                            -0.0,  1.4, -2., -1.,  1.6, #front left
                             0.0,  1.4, -2.,  1.,  1.6]) #front right
                                  ###inner### ###outer###

    q_standing = np.array([0., 0., 0.49, 0.0, 0., 0., 1.0, 
                             0.0,  .2,  0.4, -.2, -0.6,  #back left
                            -0.0, -.2,  0.4,  .2, -0.6, #back right
                            -0.0,  .2, -0.4, -.2,  0.6, #front left
                             0.0,  .2, -0.4,  .2,  0.6]) #front right
                                  ###inner### ###outer###



    # Create the contact sequence
    contact_sequence = robotoc.ContactSequence(robot)
    mu = 1.0
    friction_coefficients = {'FrontLeft_paw': mu, 'BackLeft_paw': mu, 'FrontRight_paw': mu, 'BackRight_paw': mu} 

    robot.forward_kinematics(q_standing)
    x3d0_LF = robot.frame_position('FrontLeft_paw')
    x3d0_LH = robot.frame_position('BackLeft_paw')
    x3d0_RF = robot.frame_position('FrontRight_paw')
    x3d0_RH = robot.frame_position('BackRight_paw')
    contact_positions_standing = {'FrontLeft_paw': x3d0_LF, 'BackLeft_paw': x3d0_LH, 'FrontRight_paw': x3d0_RF, 'BackRight_paw': x3d0_RH}

    print("contact_positions_standing: ", contact_positions_standing)


    ### better initial guesss using TOWR ###
    optjump_towr = OptJumpTowr()
    optjump_towr.set_initial_base_state(q_standing[0:3], np.zeros(3)) #make optjump take in quaternion
    ee_pos_towr = [x3d0_LF.copy(), x3d0_RF.copy(),x3d0_LH.copy() ,x3d0_RH.copy()]
    for ee in ee_pos_towr:
        ee[-1] = 0.0
    optjump_towr.set_initial_EE_state(ee_pos_towr)
    optjump_towr.set_takeoff_duration(take_off_duration)
    optjump_towr.set_jump_length(jump_length[0])
    optjump_towr.solve()
    flight_duration = optjump_towr.get_flight_duration()


    contact_status_standing = robot.create_contact_status()
    contact_status_standing.activate_contacts(['FrontLeft_paw', 'BackLeft_paw', 'FrontRight_paw', 'BackRight_paw'])
    contact_status_standing.set_contact_placements(contact_positions_standing)
    contact_status_standing.set_friction_coefficients(friction_coefficients)
    contact_sequence.init(contact_status_standing)

    contact_status_flying = robot.create_contact_status()
    contact_sequence.push_back(contact_status_flying, t0+take_off_duration, sto=use_sto)

    contact_status_landing = robot.create_contact_status()
    contact_status_landing.activate_contacts(['FrontLeft_paw', 'BackLeft_paw', 'FrontRight_paw', 'BackRight_paw'])
    contact_position_landing = {k:v+jump_length for k,v in contact_positions_standing.items()}
    contact_status_landing.set_contact_placements(contact_position_landing)
    contact_status_landing.set_friction_coefficients(friction_coefficients)

    contact_sequence.push_back(contact_status_landing, t0+take_off_duration+flight_duration, sto=use_sto)

    # you can check the contact sequence via 
    # print(contact_sequence)
    T = t0 + take_off_duration + flight_duration + touch_down_duration
    N = math.floor(T/dt) 
    td = robotoc.TimeDiscretization(T,N,0)
    td.discretize(contact_sequence,t0)
    td.correct_time_steps(contact_sequence,t0)
    q_traj_takeoff = []
    q_traj_landing = []
    q_traj_flight = []


    configuration_towr = []
    theta_guess_0 = np.array([ 0.0,  0.8,  1.2, -.8, -1.2,  #back left
                              -0.0, -0.8,  1.2,  .8, -1.2, #back right
                              -0.0,  0.8, -1.2, -.8,  1.2, #front left
                               0.0,  0.8, -1.2,  .8,  1.2]) #front right
                                  ###inner###  ###outer###

    theta_guess = theta_guess_0
    for i in range(len(td)):

        grid = td[i]
        t = grid.t
        phase = grid.phase
        basepose,_ = optjump_towr.get_base_state(t)
        contact_status = contact_sequence.contact_status(phase)
        if phase == 1:
            theta_guess = theta_guess_0
        theta = calcualate_joint_states(robot,contact_status,basepose,theta_guess)

        if np.any(theta < lower_limits) or np.any(theta > upper_limits):
            print("Joint limits violated")

        theta_guess = theta
        configuration_towr.append(np.concatenate((basepose,theta)))
        if phase == 0:
            q_traj_takeoff.append(configuration_towr[-1].copy())
        elif phase == 1:
            q_traj_flight.append(configuration_towr[-1].copy())
        elif phase == 2 and grid.type != robotoc.GridType.Impact:
            q_traj_landing.append(configuration_towr[-1].copy())

    u_standing,_,_ = ID(robot,q_traj_takeoff[0],np.zeros(robot.dimv()),np.zeros(robot.dimv()),contact_status_standing,joint_efforts_limit)

    # Create the constraints
    constraints           = robotoc.Constraints(barrier_param=1.0e-06, fraction_to_boundary_rule=0.995)
    joint_position_lower  = robotoc.JointPositionLowerLimit(robot)
    joint_position_upper  = robotoc.JointPositionUpperLimit(robot)
    joint_velocity_lower  = robotoc.JointVelocityLowerLimit(robot)
    joint_velocity_upper  = robotoc.JointVelocityUpperLimit(robot)
    joint_torques_lower   = robotoc.JointTorquesLowerLimit(robot)
    joint_torques_upper   = robotoc.JointTorquesUpperLimit(robot)
    friction_cone         = robotoc.FrictionCone(robot)
    constraints.add("joint_position_lower", joint_position_lower)
    constraints.add("joint_position_upper", joint_position_upper)
    constraints.add("joint_velocity_lower", joint_velocity_lower)
    constraints.add("joint_velocity_upper", joint_velocity_upper)
    constraints.add("joint_torques_lower", joint_torques_lower)
    constraints.add("joint_torques_upper", joint_torques_upper)
    constraints.add("friction_cone", friction_cone)

    # Create the STO cost function. This is necessary even empty one to construct an OCP with a STO problem
    sto_cost = robotoc.STOCostFunction()
    # Create the STO constraints 
    sto_constraints = robotoc.STOConstraints(minimum_dwell_times=[0.10, 0.15,touch_down_duration*0.7],
                                                barrier_param=1.0e-03, 
                                                fraction_to_boundary_rule=0.995)

    # Create the cost function
    cost = robotoc.CostFunction()
    refrence_traj =TrajectoryRef(robot,[q_traj_takeoff,q_traj_flight,q_traj_landing])
    #refrence_traj =TrajectoryRef(robot,[q_traj_takeoff,q_traj_flight])
    q_land = q_traj_landing[-1]
    q_weight = 10*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                            0.01, 0.01, 0.01, 0.01, 0.01, 
                            0.01, 0.01, 0.01, 0.01, 0.01,
                            0.01, 0.01, 0.01, 0.01, 0.01,
                            0.01, 0.01, 0.01, 0.01, 0.01])
    q_weight_terminal = np.ones(robot.dimv())
    q_weight_terminal[:2] = 1000*np.array([1.0, 1.0])
    q_weight_terminal[2] = 1
    q_weight_terminal[3:6] = 0.1*np.array([1.0, 1.0, 1.0])
    q_weight_terminal[6:] = 0.1*np.ones(robot.dimv()-6)
    v_weight = np.full(robot.dimv(), 1.0e-6)
    a_weight = np.full(robot.dimv(), 1.0e-06)
    q_weight_impact = np.array([0., 0., 0., 0., 0., 0., 
                            0.1, 0.1, 0.1, 0.1, 0.1, 
                            0.1, 0.1, 0.1, 0.1, 0.1,
                            0.1, 0.1, 0.1, 0.1, 0.1,
                            0.1, 0.1, 0.1, 0.1, 0.1])
    v_weight_impact = np.full(robot.dimv(), 1.0)
    dv_weight_impact = np.full(robot.dimv(), 1.0e3)
    u_weight = np.full(robot.dimu(), 1.0e-3)
    u_ref =np.zeros_like(u_standing)



    config_cost = robotoc.ConfigurationSpaceCost(robot)
    config_cost.set_ref(refrence_traj)
    #config_cost.set_q_ref(q_land)
    config_cost.set_q_weight(q_weight)
    config_cost.set_q_weight_terminal(q_weight)
    config_cost.set_q_weight_impact(q_weight_impact)
    #config_cost.set_v_weight(v_weight)
    config_cost.set_v_weight_terminal(v_weight)
    #config_cost.set_v_weight_impact(v_weight_impact)
    config_cost.set_dv_weight_impact(dv_weight_impact)
    config_cost.set_a_weight(a_weight)
    config_cost.set_u_weight(u_weight)
    config_cost.set_u_ref(u_ref)
    cost.add("config_cost", config_cost)

    # Create the OCP with the STO problem
    ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, 
                        sto_cost=sto_cost, sto_constraints=sto_constraints, 
                        contact_sequence=contact_sequence, T=T, N=N)
    # Create the OCP solver
    solver_options = robotoc.SolverOptions()
    solver_options.kkt_tol_mesh = 0.1
    solver_options.max_dt_mesh = T/N 
    solver_options.max_iter = 800
    solver_options.nthreads = 8
    solver_options.initial_sto_reg_iter = 0
    solver_options.enable_line_search=False
    solver_options.enable_benchmark=True
    solver_options.max_dts_riccati = 0.05
    ocp_solver = robotoc.OCPSolver(ocp=ocp, solver_options=solver_options)
    t=t0
    q_0 = configuration_towr[0]
    v_0 = np.zeros(robot.dimv())
    ocp_solver.discretize(t)
    td = ocp_solver.get_time_discretization()
    sol_towr = make_sol_feaseble(robot,configuration_towr,td,contact_sequence)
    ocp_solver.set_solution(sol_towr)

    #ocp_solver.set_solution("q",q_0)
    #f_init = np.array([0,0,0.25])*robot.total_weight()
    #ocp_solver.set_solution("f",f_init)
    #ocp_solver.set_solution("v",v_0)
    ocp_solver.init_constraints()


    ocp_solver.init_constraints()
    print("Initial KKT error: ", ocp_solver.KKT_error(t0, q_0, v_0))
    ocp_solver.solve(t0, q_0, v_0)
    print("KKT error after convergence: ", ocp_solver.KKT_error(t0, q_0, v_0))
    print(ocp_solver.get_solver_statistics())

    print("flight duration: ", flight_duration)




    sol = ocp_solver.get_solution()
    td = ocp_solver.get_time_discretization()
    F_contact = ocp_solver.get_solution("f", "WORLD")

    print("Number of grids: ", len(td))

    
    return ocp_solver.get_solver_statistics()


def plot_kkt(solver_stats,jump_lens,save_prefix):
    plt.figure()

    for i,solver_stat in enumerate(solver_stats):
        l = jump_lens[i]
        KKT = []
        Iter = []
        iter = 0
        for j in range(solver_stat.iter):
            KKT.append(math.log10(math.sqrt(solver_stat.performance_index[j].kkt_error)))
            Iter.append(iter)
            iter += 1
        plt.plot(Iter,KKT,label=f'jump length: {l}')
        print("final  KKT error: ", math.sqrt(solver_stat.performance_index[-1].kkt_error))
        print("final log KKT error: ", KKT[-1])
    
    plt.xlabel('Iteration')
    plt.ylabel('KKT error')
    plt.legend()
    plt.savefig(save_prefix + "_kkt.pdf")

    tikzplotlib.clean_figure()
    tikzplotlib.save(save_prefix + "_kkt.tex",externalize_tables=True,override_externals=True)



if __name__ == "__main__":
    folder_path = "/home/bolivar/OLYMPOC/robotoc/examples/olympus/python/plots/"
    save_prefix = folder_path + "convergence"
    jump_lens = [2,3,4]
    solver_stats = []
    for jump_len in jump_lens:
        solver_stats.append(test_jump(jump_len))
    plot_kkt(solver_stats,jump_lens,save_prefix)

    




        


       



   


