import robotoc
import numpy as np
import math

def solve_jump_sto_anymal(take_off_duartion, flight_duration, touch_down_duration, min_dwell_landing):
    model_info = robotoc.RobotModelInfo()
    model_info.urdf_path = '/home/bolivar/OLYMPOC/robotoc/examples/anymal/anymal_b_simple_description/urdf/anymal.urdf'
    #model_info.urdf_path = '/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympmal.urdf'
    model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
    baumgarte_time_step = 0.05
    model_info.point_contacts = [robotoc.ContactModelInfo('LF_FOOT', baumgarte_time_step),
                                 robotoc.ContactModelInfo('LH_FOOT', baumgarte_time_step),
                                 robotoc.ContactModelInfo('RF_FOOT', baumgarte_time_step),
                                 robotoc.ContactModelInfo('RH_FOOT', baumgarte_time_step)]
    robot = robotoc.Robot(model_info)
    robot.set_gravity(-3.72)

    dt = 0.02
    jump_length = np.array([0.8, 0, 0])
    t0 = 0.

    # Create the cost function
    cost = robotoc.CostFunction()
    q_standing = np.array([0., 0., 0.472, 0., 0., 0., 1.0, 
                           -0.1,  0.7, -1.0, 
                           -0.1, -0.7,  1.0, 
                            0.1,  0.7, -1.0, 
                            0.1, -0.7,  1.0])
    
    q_standing = np.array([0., 0., 0.472, 0., 0., 0., 1.0, 
                           -0.1,  0.3, -0.5, 
                           -0.1, -0.3,  0.5, 
                            0.1,  0.3, -0.5, 
                            0.1, -0.3,  0.5])
    

    q_ref = q_standing.copy()
    q_ref[0:3] += jump_length
    q_weight = np.array([1.0, 0., 0., 1.0, 1.0, 1.0, 
                         0.001, 0.001, 0.001, 
                         0.001, 0.001, 0.001,
                         0.001, 0.001, 0.001,
                         0.001, 0.001, 0.001])
    v_weight = np.full(robot.dimv(), 1.0)
    a_weight = np.full(robot.dimv(), 1.0e-06)
    q_weight_impact = np.array([0., 0., 0., 100., 100., 100., 
                          0.1, 0.1, 0.1, 
                          0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1])
    v_weight_impact = np.full(robot.dimv(), 1.0)
    dv_weight_impact = np.full(robot.dimv(), 1.0e-06)
    config_cost = robotoc.ConfigurationSpaceCost(robot)
    config_cost.set_q_ref(q_ref)
    config_cost.set_q_weight(q_weight)
    config_cost.set_q_weight_terminal(q_weight)
    config_cost.set_q_weight_impact(q_weight_impact)
    config_cost.set_v_weight(v_weight)
    config_cost.set_v_weight_terminal(v_weight)
    config_cost.set_v_weight_impact(v_weight_impact)
    config_cost.set_dv_weight_impact(dv_weight_impact)
    config_cost.set_a_weight(a_weight)
    cost.add("config_cost", config_cost)

    # Create the constraints
    constraints           = robotoc.Constraints(barrier_param=1.0e-03, fraction_to_boundary_rule=0.995)
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


    # Create the contact sequence
    contact_sequence = robotoc.ContactSequence(robot)
    mu = 0.7
    friction_coefficients = {'LF_FOOT': mu, 'LH_FOOT': mu, 'RF_FOOT': mu, 'RH_FOOT': mu} 

    robot.forward_kinematics(q_standing)
    x3d0_LF = robot.frame_position('LF_FOOT')
    x3d0_LH = robot.frame_position('LH_FOOT')
    x3d0_RF = robot.frame_position('RF_FOOT')
    x3d0_RH = robot.frame_position('RH_FOOT')
    contact_positions = {'LF_FOOT': x3d0_LF, 'LH_FOOT': x3d0_LH, 'RF_FOOT': x3d0_RF, 'RH_FOOT': x3d0_RH} 

    contact_status_standing = robot.create_contact_status()
    contact_status_standing.activate_contacts(['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT'])
    contact_status_standing.set_contact_placements(contact_positions)
    contact_status_standing.set_friction_coefficients(friction_coefficients)
    contact_sequence.init(contact_status_standing)

    contact_status_flying = robot.create_contact_status()
    contact_sequence.push_back(contact_status_flying, t0+take_off_duartion, sto=True)

    contact_positions['LF_FOOT'] += jump_length
    contact_positions['LH_FOOT'] += jump_length
    contact_positions['RF_FOOT'] += jump_length
    contact_positions['RH_FOOT'] += jump_length
    contact_status_standing.set_contact_placements(contact_positions)
    contact_sequence.push_back(contact_status_standing, t0+take_off_duartion+flight_duration, sto=True)

    # you can check the contact sequence via 
    # print(contact_sequence)

    # Create the STO cost function. This is necessary even empty one to construct an OCP with a STO problem
    sto_cost = robotoc.STOCostFunction()
    # Create the STO constraints 
    sto_constraints = robotoc.STOConstraints(minimum_dwell_times=[0.15, 0.15, min_dwell_landing],
                                             barrier_param=1.0e-03, 
                                             fraction_to_boundary_rule=0.995)

    T = t0 + take_off_duartion + flight_duration + touch_down_duration
    N = math.floor(T/dt) 
    # Create the OCP with the STO problem
    ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, 
                      sto_cost=sto_cost, sto_constraints=sto_constraints, 
                      contact_sequence=contact_sequence, T=T, N=N)
    # Create the OCP solver
    solver_options = robotoc.SolverOptions()
    solver_options.kkt_tol_mesh = 1.0
    solver_options.max_dt_mesh = T/N 
    solver_options.max_iter = 200
    solver_options.nthreads = 4
    ocp_solver = robotoc.OCPSolver(ocp=ocp, solver_options=solver_options)

    # Initial time and intial state 
    t = 0.
    q = q_standing
    v = np.zeros(robot.dimv())

    ocp_solver.discretize(t)
    ocp_solver.set_solution("q", q)
    ocp_solver.set_solution("v", v)
    f_init = np.array([0.0, 0.0, 0.25*robot.total_weight()])
    ocp_solver.set_solution("f", f_init)

    ocp_solver.init_constraints()
    ocp_solver.solve(t, q, v)
    return ocp_solver.KKT_error(t, q, v)



def solve_jump_sto(take_off_duartion, flight_duration, touch_down_duration, min_dwell_landing):
    model_info = robotoc.RobotModelInfo()
    model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympmal.urdf"
    model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
    baumgarte_time_step = 0.05
    model_info.point_contacts = [robotoc.ContactModelInfo('FrontLeft_paw', baumgarte_time_step),
                                 robotoc.ContactModelInfo('BackLeft_paw', baumgarte_time_step),
                                 robotoc.ContactModelInfo('FrontRight_paw', baumgarte_time_step),
                                 robotoc.ContactModelInfo('BackRight_paw', baumgarte_time_step)]
    robot = robotoc.Robot(model_info)
    robot.set_lower_joint_position_limit(np.full(robot.dimv()-6, -2.0))
    robot.set_upper_joint_position_limit(np.full(robot.dimv()-6, 2.0))
    robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
    robot.set_joint_effort_limit(np.full(robot.dimv()-6, 25.0))


    dt = 0.02
    jump_length = np.array([0.5, 0, 0])
    t0 = 0.

    # Create the cost function
    cost = robotoc.CostFunction()
    ##q_standing = np.array([0., 0., 0.4, 0., 0., 0., 1.0, 
    ##                    0.0,  0.4,  0.8, 
    ##                    0.0, -0.4,  0.8, 
    ##                    0.0,  0.4, -0.8, 
    ##                    0.0,  0.4, -0.8])


    q_standing = np.array([0., 0., 0.472, 0., 0., 0., 1.0, 
                           -0.0,  0.1, -0.1, 
                           -0.0, -0.1,  0.1, 
                            0.0,  0.1, -0.1, 
                            0.0, -0.1,  0.1])
    

    q_ref = q_standing.copy()
    q_ref[0:3] += jump_length
    q_weight = np.array([100, 0., 0., 1.0, 0.001, 1.0, 
                         0.001, 0.001, 0.001, 
                         0.001, 0.001, 0.001,
                         0.001, 0.001, 0.001,
                         0.001, 0.001, 0.001])
    v_weight = np.full(robot.dimv(), 0)
    a_weight = np.full(robot.dimv(), 1.0e-06)
    q_weight_impact = np.array([0., 0., 0., 100., 100., 100., 
                          0.1, 0.1, 0.1, 
                          0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1])
    v_weight_impact = np.full(robot.dimv(), 1.0)
    dv_weight_impact = np.full(robot.dimv(), 1.0e-06)
    config_cost = robotoc.ConfigurationSpaceCost(robot)
    config_cost.set_q_ref(q_ref)
    config_cost.set_q_weight(q_weight)
    config_cost.set_q_weight_terminal(q_weight)
    config_cost.set_q_weight_impact(q_weight_impact)
    config_cost.set_v_weight(v_weight)
    config_cost.set_v_weight_terminal(v_weight)
    config_cost.set_v_weight_impact(v_weight_impact)
    config_cost.set_dv_weight_impact(dv_weight_impact)
    config_cost.set_a_weight(a_weight)
    cost.add("config_cost", config_cost)

    # Create the constraints
    constraints           = robotoc.Constraints(barrier_param=1.0e-03, fraction_to_boundary_rule=0.995)
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


    # Create the contact sequence
    contact_sequence = robotoc.ContactSequence(robot)
    mu = 0.7
    friction_coefficients = {'FrontLeft_paw': mu, 'BackLeft_paw': mu, 'FrontRight_paw': mu, 'BackRight_paw': mu} 

    robot.forward_kinematics(q_standing)
    x3d0_LF = robot.frame_position('FrontLeft_paw')
    x3d0_LH = robot.frame_position('BackLeft_paw')
    x3d0_RF = robot.frame_position('FrontRight_paw')
    x3d0_RH = robot.frame_position('BackRight_paw')
    contact_positions = {'FrontLeft_paw': x3d0_LF, 'BackLeft_paw': x3d0_LH, 'FrontRight_paw': x3d0_RF, 'BackRight_paw': x3d0_RH} 


    contact_status_standing = robot.create_contact_status()
    contact_status_standing.activate_contacts(['FrontLeft_paw', 'BackLeft_paw', 'FrontRight_paw', 'BackRight_paw'])
    contact_status_standing.set_contact_placements(contact_positions)
    contact_status_standing.set_friction_coefficients(friction_coefficients)
    contact_sequence.init(contact_status_standing)

    contact_status_flying = robot.create_contact_status()
    contact_sequence.push_back(contact_status_flying, t0+take_off_duartion, sto=True)

    contact_positions['FrontLeft_paw'] += jump_length
    contact_positions['BackLeft_paw'] += jump_length
    contact_positions['FrontRight_paw'] += jump_length
    contact_positions['BackRight_paw'] += jump_length
    contact_status_standing.set_contact_placements(contact_positions)
    contact_sequence.push_back(contact_status_standing, t0+take_off_duartion+flight_duration, sto=True)

    # you can check the contact sequence via 
    # print(contact_sequence)

    # Create the STO cost function. This is necessary even empty one to construct an OCP with a STO problem
    sto_cost = robotoc.STOCostFunction()
    # Create the STO constraints 
    sto_constraints = robotoc.STOConstraints(minimum_dwell_times=[0.15, 0.15, min_dwell_landing],
                                             barrier_param=1.0e-03, 
                                             fraction_to_boundary_rule=0.995)

    T = t0 + take_off_duartion + flight_duration + touch_down_duration
    N = math.floor(T/dt) 
    # Create the OCP with the STO problem
    ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, 
                      sto_cost=sto_cost, sto_constraints=sto_constraints, 
                      contact_sequence=contact_sequence, T=T, N=N)
    # Create the OCP solver
    solver_options = robotoc.SolverOptions()
    solver_options.kkt_tol_mesh = 0.1
    solver_options.max_dt_mesh = T/N 
    solver_options.max_iter = 200
    solver_options.nthreads = 4
    solver_options.initial_sto_reg_iter =20
    ocp_solver = robotoc.OCPSolver(ocp=ocp, solver_options=solver_options)

    # Initial time and intial state 
    t = 0.
    q = q_standing
    v = np.zeros(robot.dimv())

    ocp_solver.discretize(t)
    ocp_solver.set_solution("q", q)
    ocp_solver.set_solution("v", v)
    f_init = np.array([0.0, 0.0, 0.25*robot.total_weight()])
    ocp_solver.set_solution("f", f_init)

    ocp_solver.init_constraints()
    #print("Initial KKT error: ", ocp_solver.KKT_error(t, q, v))
    ocp_solver.solve(t, q, v)
    kkt_error = ocp_solver.KKT_error(t, q, v)
    #print("KKT error after convergence: ", kkt_error)
    return kkt_error



if __name__ == "__main__":

    take_off_duration = np.linspace(0.7,0.9,3)
    flight_duration = np.linspace(0.2,0.4,3)
    touch_down_duration = np.linspace(0.7,0.9,3)
    min_dwell_time_fraq = np.linspace(0.5,0.9,3)
    N_BEST = 15

    best_results = []



    for x in take_off_duration:
        for y in flight_duration:
            for z in touch_down_duration:
                for w in min_dwell_time_fraq:
                
                    kkt_error = solve_jump_sto(x,y,z,z*w)

                    best_results.append((kkt_error, x,y,z,w))
                    if len(best_results) > N_BEST:
                        best_results = sorted(best_results, key=lambda x: x[0])
                        best_results.pop()


                    print(f"take_off_duration: {x:.2f}, flight_duration: {y:.2f}, touch_down_duration: {z:.2f}, min_dwell_time_fraq: {w:.2f}, kkt_error: {kkt_error}")


    print(" ========================== Best results ===========================")
    print("")
    i = 1
    for kkt_error, x, y, z, w in best_results:
        print(f"{i} take_off_duration: {x:.2f}, flight_duration: {y:.2f}, touch_down_duration: {z:.3f}, minimum_dwell_time_fraq: {w:.3f}, kkt_error: {kkt_error}")
        i += 1
    


