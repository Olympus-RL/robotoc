import robotoc
import numpy as np
import math


model_info = robotoc.RobotModelInfo()
model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus_open.urdf"
model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
baumgarte_time_step = 0.05
model_info.point_contacts = [robotoc.ContactModelInfo('FrontLeft_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('BackLeft_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('FrontRight_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('BackRight_paw', baumgarte_time_step)]
robot = robotoc.Robot(model_info)
robot.set_gravity(-3.72)
robot.set_lower_joint_position_limit(np.full(robot.dimv()-6, -2.0))
robot.set_upper_joint_position_limit(np.full(robot.dimv()-6, 2.0))
robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
robot.set_joint_effort_limit(np.full(robot.dimv()-6, 30.0))

dt = 0.02
take_off_vel= np.array([0, 0,2.0])
take_off_duration = 0.3
flight_duration = 0.1
t0 = 0.

# Create the cost function
cost = robotoc.CostFunction()
q_standing = np.array([0., 0., 0.4, 0., 0., 0., 1.0, 
                         0.0,  1.4,  1.8,  #back left
                        -0.0, -1.4,  1.8, #back right
                        -0.0,  1.4, -1.8, #front left
                         0.0,  1.4, -1.8]) #front right
q_ref = np.zeros(robot.dimq())
q_ref[2] = 2.0
q_weight = np.zeros(robot.dimv())
q_weight[:3] = np.array([10.0, 10.0, 100.0])

config_cost = robotoc.ConfigurationSpaceCost(robot)
config_cost.set_q_ref(q_ref)
config_cost.set_q_weight_terminal(q_weight)
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
contact_sequence.push_back(contact_status_flying, t0+take_off_duration, sto=True)


# you can check the contact sequence via 
# print(contact_sequence)

# Create the STO cost function. This is necessary even empty one to construct an OCP with a STO problem
sto_cost = robotoc.STOCostFunction()
# Create the STO constraints 
sto_constraints = robotoc.STOConstraints(minimum_dwell_times=[0.05, 0.05],
                                            barrier_param=1.0e-03, 
                                            fraction_to_boundary_rule=0.995)

T = t0 + take_off_duration + flight_duration 
N = math.floor(T/dt) 
# Create the OCP with the STO problem
ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, 
                    sto_cost=sto_cost, sto_constraints=sto_constraints, 
                    contact_sequence=contact_sequence, T=T, N=N)
# Create the OCP solver
solver_options = robotoc.SolverOptions()
solver_options.kkt_tol_mesh = 1.0
solver_options.max_dt_mesh = T/N 
solver_options.max_iter = 400
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
print("Initial KKT error: ", ocp_solver.KKT_error(t, q, v))
ocp_solver.solve(t, q, v)
print("KKT error after convergence: ", ocp_solver.KKT_error(t, q, v))
print(ocp_solver.get_solver_statistics())

exit()


# Plot results
kkt_data = [math.sqrt(e.kkt_error) for e in ocp_solver.get_solver_statistics().performance_index] + [ocp_solver.KKT_error()] # append KKT after convergence
ts_data = ocp_solver.get_solver_statistics().ts + [contact_sequence.event_times()] # append ts after convergence

plot_ts = robotoc.utils.PlotConvergence()
plot_ts.ylim = [0., 1.8]
plot_ts.plot(kkt_data=kkt_data, ts_data=ts_data, fig_name='jump_sto', 
             save_dir='jump_sto_log')

plot_f = robotoc.utils.PlotContactForce(mu=mu)
plot_f.plot(f_traj=ocp_solver.get_solution('f', 'WORLD'), 
            time_discretization=ocp_solver.get_time_discretization(), 
            fig_name='jump_sto_f', save_dir='jump_sto_log')

# Display results
viewer = robotoc.utils.TrajectoryViewer(model_info=model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=mu)

Q = ocp_solver.get_solution('q')
F = ocp_solver.get_solution('f', 'WORLD')



viewer.display(ocp_solver.get_time_discretization(), 
               ocp_solver.get_solution('q'), 
               ocp_solver.get_solution('f', 'WORLD'))