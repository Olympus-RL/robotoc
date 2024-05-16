import robotoc
from calculate_joint_states import calcualate_joint_states
from generate_feasble_trajectory import generate_feaseble_trajectory
import numpy as np
import math
import copy


model_info = robotoc.RobotModelInfo()
model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus_open.urdf"
model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
model_info.contact_inv_damping = 1e-6;

baumgarte_time_step = 0.05
model_info.point_contacts = [robotoc.ContactModelInfo('FrontLeft_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('BackLeft_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('FrontRight_paw', baumgarte_time_step),
                             robotoc.ContactModelInfo('BackRight_paw', baumgarte_time_step)]
robot = robotoc.Robot(model_info)
#robot.set_gravity(-3.72)
robot.set_lower_joint_position_limit(np.array([
    -0.5, -2.0, -2.0,
    -0.5, -2.0, -2.0,
    -0.5, -2.0, -2.0,
    -0.5, -2.0, -2.0,
]))
robot.set_upper_joint_position_limit(np.array([
    0.5,  2.0, 2.0,
    0.5,  2.0, 2.0,
    0.5,  2.0, 2.0,
    0.5,  2.0, 2.0,
]))

robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
robot.set_joint_effort_limit(np.full(robot.dimv()-6, 24.8))

dt = 0.02
jump_length = np.array([0.0, 0, 0])
standing_time = 0.5
t0 = 0.

# Create the cost function
q_standing = np.array([0., 0., 0.49, 0., 0., 0., 1.0, 
                         0.0,  0.3,  0.4,  #back left
                        -0.0, -0.3,  0.4, #back right
                        -0.0,  0.3, -0.4, #front left
                         0.0,  0.3, -0.4]) #front right
theta_guess_0 = np.array( [-0.0,  1.0,  1.2, 
                           -0.0, -1.0,  1.2, 
                            0.0,  1.0, -1.2, 
                            0.0,  1.0, -1.2]) #front right)


# Create the contact sequence
contact_sequence = robotoc.ContactSequence(robot)
mu = 0.7
friction_coefficients = {'FrontLeft_paw': mu, 'BackLeft_paw': mu, 'FrontRight_paw': mu, 'BackRight_paw': mu} 

robot.forward_kinematics(q_standing)
x3d0_LF = robot.frame_position('FrontLeft_paw')
x3d0_LH = robot.frame_position('BackLeft_paw')
x3d0_RF = robot.frame_position('FrontRight_paw')
x3d0_RH = robot.frame_position('BackRight_paw')
contact_positions_standing = {'FrontLeft_paw': x3d0_LF, 'BackLeft_paw': x3d0_LH, 'FrontRight_paw': x3d0_RF, 'BackRight_paw': x3d0_RH}

contact_status_standing = robot.create_contact_status()
contact_status_standing.activate_contacts(['FrontLeft_paw', 'BackLeft_paw', 'FrontRight_paw', 'BackRight_paw'])
contact_status_standing.set_contact_placements(contact_positions_standing)
contact_status_standing.set_friction_coefficients(friction_coefficients)
contact_sequence.init(contact_status_standing)

pose_squatting = np.array([0., 0., 0.30, 0., 0., 0., 1.0])
theta_squating = calcualate_joint_states(robot,contact_positions_standing,pose_squatting,theta_guess_0)
q_squating = np.concatenate((pose_squatting,theta_squating))

cost = robotoc.CostFunction()
q_ref = q_squating.copy()
q_weight = np.array([1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 
                        0.001, 0.001, 0.001, 
                        0.001, 0.001, 0.001,
                        0.001, 0.001, 0.001,
                        0.001, 0.001, 0.001])
v_weight = np.full(robot.dimv(), 1.0e-03)
a_weight = np.full(robot.dimv(), 1.0e-06)
config_cost = robotoc.ConfigurationSpaceCost(robot)
config_cost.set_q_ref(q_ref)
config_cost.set_q_weight(q_weight)
config_cost.set_q_weight_terminal(q_weight)
config_cost.set_v_weight(v_weight)
config_cost.set_v_weight_terminal(v_weight)
config_cost.set_a_weight(a_weight)
cost.add("config_cost", config_cost)

# Create the constraints
constraints           = robotoc.Constraints(barrier_param=1.0e-03, fraction_to_boundary_rule=0.95)
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


T = t0 + standing_time
N = math.floor(T/dt) 
# Create the OCP with the STO problem
ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, 
                    contact_sequence=contact_sequence, T=T, N=N)
# Create the OCP solver
solver_options = robotoc.SolverOptions()
solver_options.kkt_tol_mesh = 0.1
solver_options.max_dt_mesh = T/N 
solver_options.max_iter = 200
solver_options.nthreads = 4
solver_options.initial_sto_reg_iter = 0
solver_options.max_dts_riccati = 0.05
ocp_solver = robotoc.OCPSolver(ocp=ocp, solver_options=solver_options)

t=t0

ocp_solver.set_solution("q",q_standing)
f_init = np.array([0,0,0.25])*robot.total_weight()
ocp_solver.set_solution("f",f_init)
ocp_solver.set_solution("v",np.zeros(robot.dimv()))
ocp_solver.discretize(t)






# Initial time and intial state 
q = q_standing
v = np.zeros(robot.dimv())

theta_guess = theta_guess_0

q_traj =[]
td = ocp_solver.get_time_discretization()
dts = []


for i in range(len(td)):
    grid = td[i]
    dts.append(grid.dt)
    h = q_standing[2] #+ (q_squating[2] - q_standing[2])/T * grid.t
    q_i = q_squating.copy()
    q_i[2] = h
    pose_i = q_i[:7]
    theta_i = calcualate_joint_states(robot,contact_positions_standing,pose_i,theta_guess)
    theta_guess = theta_i
    q_i = np.concatenate((pose_i,theta_i))
    q_traj.append(q_i)


dts.pop()

solution_guess = generate_feaseble_trajectory(robot,q_traj,dts,contact_positions_standing, mu)

# check if the solution is feasible
for i in range(len(solution_guess)):
    grid = td[i]
    # state eq
    if grid.stage != robotoc.grid_info.GridType.Terminal:
        robot.set

ocp_solver.set_solution(solution_guess)
ocp_solver.init_constraints()
print("Initial KKT error: ", ocp_solver.KKT_error(t, q, v))
ocp_solver.solve(t0, q, v)
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

# Display results
viewer = robotoc.utils.TrajectoryViewer(model_info=model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=mu)

Q = ocp_solver.get_solution('q')
F = ocp_solver.get_solution('f', 'WORLD')



viewer.display(ocp_solver.get_time_discretization(), 
               ocp_solver.get_solution('q'), 
               ocp_solver.get_solution('f', 'WORLD'))