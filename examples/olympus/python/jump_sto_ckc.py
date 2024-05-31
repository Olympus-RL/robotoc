import robotoc
import numpy as np
import math

from initialization import calcualate_joint_states


model_info = robotoc.RobotModelInfo()
model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus.urdf"
model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
baumgarte_time_step = 0.05
model_info.point_contacts = [robotoc.ContactModelInfo('FrontLeft_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('BackLeft_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('FrontRight_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('BackRight_paw', baumgarte_time_step)]
model_info.ckcs = [robotoc.CKCInfo( "FrontLeft_ankle_outer","FrontLeft_ankle_inner",baumgarte_time_step),
                    robotoc.CKCInfo("FrontRight_ankle_outer","FrontRight_ankle_inner",baumgarte_time_step),
                    robotoc.CKCInfo("BackLeft_ankle_outer","BackLeft_ankle_inner",baumgarte_time_step),
                    robotoc.CKCInfo("BackRight_ankle_outer","BackRight_ankle_inner",baumgarte_time_step)]

robot = robotoc.Robot(model_info)
robot.set_lower_joint_position_limit(np.full(robot.dimv()-6, -2.0))
robot.set_upper_joint_position_limit(np.full(robot.dimv()-6, 2.0))
robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
joint_efforts_limit = np.full(robot.dimv()-6, 24.0)
knee_idx = []
idx =0
for i in range(4):
    knee_idx.append(i*5+2)
    knee_idx.append(i*5+4)
#joint_efforts_limit[knee_idx] = 0.0001
robot.set_joint_effort_limit(joint_efforts_limit)
robot.set_gravity(-3.72)


dt = 0.02
jump_length = np.array([0.5, 0, 0])
take_off_duration = 0.5
flight_duration = 0.6
touch_down_duration = 0.8
t0 = 0.






# Create the cost function
cost = robotoc.CostFunction()
q_standing = np.array([0., 0., 0.5, 0.0, 0., 0., 1.0, 
                         0.0,  0.2,  .4, -.2, -.4,  #back left
                        -0.0, -0.2,  .4,  .2, -.4, #back right
                        -0.0,  0.2, -.4, -.2,  .4, #front left
                         0.0,  0.2, -.4,  .2,  .4]) #front right
                              ###inner###  ###outer###



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
contact_sequence.push_back(contact_status_flying, t0+take_off_duration, sto=False)

contact_positions['FrontLeft_paw'] += jump_length
contact_positions['BackLeft_paw'] += jump_length
contact_positions['FrontRight_paw'] += jump_length
contact_positions['BackRight_paw'] += jump_length
contact_status_standing.set_contact_placements(contact_positions)
contact_sequence.push_back(contact_status_standing, t0+take_off_duration+flight_duration, sto=False)


theta = q_standing[7:].copy()
q = q_standing.copy()
pose = q_standing[:7].copy()
theta = calcualate_joint_states(robot, contact_status_standing,pose, theta)
q[:7] = pose
q[7:] = theta
q_standing = q.copy()


q_ref = q_standing.copy()
q_ref[0:3] += jump_length
q_weight = np.array([1.0, 0., 0., 1.0, 1.0, 1.0, 
                        0.00001, 0.00001,0.00001, 0.00001, 0.00001, 
                        0.00001, 0.00001,0.00001, 0.00001, 0.00001,
                        0.00001, 0.00001,0.00001, 0.00001, 0.00001,
                        0.00001, 0.00001,0.00001, 0.00001, 0.00001])
v_weight = np.full(robot.dimv(), 1.0)
a_weight = np.full(robot.dimv(), 1.0e-06)
q_weight_impact = np.array([0., 0., 0., 100., 100., 100., 
                        0.1,0.1,0.1, 0.1, 0.1, 
                        0.1,0.1,0.1, 0.1, 0.1,
                        0.1,0.1,0.1, 0.1, 0.1,
                        0.1,0.1,0.1, 0.1, 0.1])
print(q_weight_impact.shape)
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







# you can check the contact sequence via 
# print(contact_sequence)

# Create the STO cost function. This is necessary even empty one to construct an OCP with a STO problem
sto_cost = robotoc.STOCostFunction()
# Create the STO constraints 
sto_constraints = robotoc.STOConstraints(minimum_dwell_times=[0.10, 0.3, 0.65],
                                            barrier_param=1.0e-03, 
                                            fraction_to_boundary_rule=0.995)

T = t0 + take_off_duration + flight_duration + touch_down_duration
N = math.floor(T/dt) 
# Create the OCP with the STO problem
ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, 
                    sto_cost=sto_cost, sto_constraints=sto_constraints, 
                    contact_sequence=contact_sequence, T=T, N=N)
# Create the OCP solver
solver_options = robotoc.SolverOptions()
solver_options.kkt_tol_mesh = 1.0
solver_options.max_dt_mesh = T/N 
solver_options.max_iter = 800
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


# Plot results
#kkt_data = [math.sqrt(e.kkt_error) for e in ocp_solver.get_solver_statistics().performance_index] + [ocp_solver.KKT_error()] # append KKT after convergence
#ts_data = ocp_solver.get_solver_statistics().ts + [contact_sequence.event_times()] # append ts after convergence
#
#plot_ts = robotoc.utils.PlotConvergence()
#plot_ts.ylim = [0., 1.8]
#plot_ts.plot(kkt_data=kkt_data, ts_data=ts_data, fig_name='jump_sto', 
#             save_dir='jump_sto_log')
#
#plot_f = robotoc.utils.PlotContactForce(mu=mu)
#plot_f.plot(f_traj=ocp_solver.get_solution('f', 'WORLD'), 
#            time_discretization=ocp_solver.get_time_discretization(), 
#            fig_name='jump_sto_f', save_dir='jump_sto_log')
#
# Display results
viewer = robotoc.utils.TrajectoryViewer(model_info=model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=mu)

Q = ocp_solver.get_solution('q')
F = ocp_solver.get_solution('f', 'WORLD')



viewer.display(ocp_solver.get_time_discretization(), 
               ocp_solver.get_solution('q'), 
               ocp_solver.get_solution('f', 'WORLD'))