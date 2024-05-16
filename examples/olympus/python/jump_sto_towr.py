import robotoc
from tojr import OptJumpTowr
from calculate_joint_states import calcualate_joint_states
from generate_feasble_trajectory import generate_feaseble_trajectory
import numpy as np
import math
import copy


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
robot.set_lower_joint_position_limit(np.array([
    -0.5, -1.57, -2.0,
    -0.5, -1.57, -2.0,
    -0.5, -1.57, -2.0,
    -0.5, -1.57, -2.0,
]))
robot.set_upper_joint_position_limit(np.array([
    0.5,  1.57, 2.0,
    0.5,  1.57, 2.0,
    0.5,  1.57, 2.0,
    0.5,  1.57, 2.0,
])

)
robot.set_joint_velocity_limit(np.full(robot.dimv()-6, 31.0))
robot.set_joint_effort_limit(np.full(robot.dimv()-6, 30.0))

dt = 0.04
jump_length = np.array([0.8, 0, 0])
take_off_duration = 0.5
flight_duration = 0.20
touch_down_duration = take_off_duration
t0 = 0.

q_standing = np.array([0., 0., 0.37, 0., 0., 0., 1.0, 
                         0.0,  0.7,  1.4,  #back left
                        -0.0, -0.7,  1.4, #back right
                        -0.0,  0.7, -1.4, #front left
                         0.0,  0.7, -1.4]) #front right

# Create the cost function
cost = robotoc.CostFunction()


q_ref = q_standing.copy()
q_ref[0:3] += jump_length
q_weight = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 
                        0.001, 0.001, 0.001, 
                        0.001, 0.001, 0.001,
                        0.001, 0.001, 0.001,
                        0.001, 0.001, 0.001])
v_weight = np.full(robot.dimv(), 1.0)
a_weight = np.full(robot.dimv(), 1.0e-06)
q_weight_impact = np.array([0., 0., 0., 0., 0., 0., 
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

###




contact_status_standing = robot.create_contact_status()
contact_status_standing.activate_contacts(['FrontLeft_paw', 'BackLeft_paw', 'FrontRight_paw', 'BackRight_paw'])
contact_status_standing.set_contact_placements(contact_positions_standing)
contact_status_standing.set_friction_coefficients(friction_coefficients)
contact_sequence.init(contact_status_standing)

contact_status_flying = robot.create_contact_status()
contact_sequence.push_back(contact_status_flying, t0+take_off_duration, sto=True)

contat_position_landing = copy.deepcopy(contact_positions_standing)
contact_positions_standing['FrontLeft_paw'] += jump_length
contact_positions_standing['BackLeft_paw'] += jump_length
contact_positions_standing['FrontRight_paw'] += jump_length
contact_positions_standing['BackRight_paw'] += jump_length
contact_status_standing.set_contact_placements(contact_positions_standing)
contact_sequence.push_back(contact_status_standing, t0+take_off_duration+flight_duration, sto=True)

# you can check the contact sequence via 
# print(contact_sequence)

# Create the STO cost function. This is necessary even empty one to construct an OCP with a STO problem
sto_cost = robotoc.STOCostFunction()
# Create the STO constraints 
sto_constraints = robotoc.STOConstraints(minimum_dwell_times=[take_off_duration*0.7, flight_duration*0.3,touch_down_duration*0.7],
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
solver_options.kkt_tol_mesh = 0.1
solver_options.max_dt_mesh = T/N 
solver_options.max_iter = 100
solver_options.nthreads = 4
solver_options.initial_sto_reg_iter = 0
#solver_options.max_dts_riccati = 0.05
ocp_solver = robotoc.OCPSolver(ocp=ocp, solver_options=solver_options)
t=t0
ocp_solver.discretize(t)
ocp_solver.set_solution("q",q_standing)

dummy_sol = ocp_solver.get_solution()


# Initial time and intial state 
q = q_standing
v = np.zeros(robot.dimv())
theta_guess_0 = np.array( [-0.0,  1.0,  1.5, 
                           -0.0, -1.0,  1.5, 
                            0.0,  1.0, -1.5, 
                            0.0,  1.0, -1.5]) #front right)
theta_guess = theta_guess_0

time_discretization = ocp_solver.get_time_discretization()
con_pos_stand = {fn:robot.frame_position(fn) for fn in ['BackLeft_paw', 'BackRight_paw','FrontLeft_paw','FrontRight_paw']}
con_pos_land = {fn:robot.frame_position(fn)+jump_length for fn in ['BackLeft_paw', 'BackRight_paw','FrontLeft_paw','FrontRight_paw']}
dts_takeoff = []
dts_flight = []
dts_landing = []
q_traj_takeoff = []
q_traj_flight = []
q_traj_landing = []

for i in range(len(time_discretization)):
    grid = time_discretization[i]
    base_pose_towr, base_vel_towr = optjump_towr.get_base_state(grid.t)
    theta = q_standing[7:]
    theta_dot = np.zeros(robot.dimv()-6)
    if  grid.t < take_off_duration:
        theta= calcualate_joint_states(robot, con_pos_stand, base_pose_towr,theta_guess)
        q_traj_takeoff.append(np.concatenate((base_pose_towr,theta)))
        dts_takeoff.append(grid.dt)
        theta_guess = theta
    elif grid.t < take_off_duration + flight_duration:
        if grid.t == take_off_duration:
            print("takeoff")
            print(time_discretization[i-1])
            print(grid)
            print(time_discretization[i+1])
        theta = theta_guess_0
        theta_guess = theta_guess_0
        q_traj_flight.append(np.concatenate((base_pose_towr,theta)))
        dts_flight.append(grid.dt)
        
    else:
        if grid.type == robotoc.grid_info.GridType.Impact:
            print("landing")
            print(grid)
            print(time_discretization[i+1])
        else:
            theta= calcualate_joint_states(robot, con_pos_land, base_pose_towr,theta_guess)
            theta_guess = theta
            q_traj_landing.append(np.concatenate((base_pose_towr,theta)))
            dts_landing.append(grid.dt)


dts_takeoff.pop()
dts_flight.pop()        
dts_landing.pop()
solution_guess_takeoff = generate_feaseble_trajectory(robot,q_traj_takeoff,dts_takeoff,contact_positions_standing, mu)
solution_guess_landing = generate_feaseble_trajectory(robot,q_traj_landing,dts_landing,con_pos_land, mu)
s_impact = robotoc.SplitSolution()
s_impact.q = q_traj_landing[0]
s_impact.v = solution_guess_landing[0].v
solution_guess_landing.insert(0,s_impact)
vel_flight = []
acc_flight = []
solution_guess_flight = []
for i, dt in enumerate(dts_flight):
    v = robot.subtract_configuration(q_traj_flight[i+1], q_traj_flight[i]) / dt
    vel_flight.append(v)
vel_flight.append(vel_flight[-1].copy())
for i, dt in enumerate(dts_flight):
    a = (vel_flight[i+1] - vel_flight[i]) / dt
    acc_flight.append(a)
acc_flight.append(acc_flight[-1].copy())

for i,(q,v,a) in enumerate(zip(q_traj_flight,vel_flight,acc_flight)):
    s = robotoc.SplitSolution(robot)
    s.q = q
    s.v = v
    s.a = a
    solution_guess_flight.append(s)


solution_towr = solution_guess_takeoff + solution_guess_flight + solution_guess_landing


print("len(solution_towr): ", len(solution_towr))
print("len(dummy_sol): ", len(dummy_sol))

test = dummy_sol
if True:
    for s_test,s_towr in zip(test,solution_towr):
        s_test.q = s_towr.q
        s_test.v = s_towr.v
        s_test.a = s_towr.a
        s_test.u = s_towr.u
        s_test.f = s_towr.f
        #s_test.lmd = np.random.randn(*s_test.lmd.shape)
        #s_test.gmm = np.random.randn(*s_test.gmm.shape)
        #s_test.beta = np.random.randn(*s_test.beta.shape)
        #s_test.mu = [np.random.randn(*m.shape) for m in s_test.mu]
        #s_test.nu_passive = np.random.randn(*s_test.nu_passive.shape)

        s_test.set_f_stack()
        s_test.set_mu_stack()




ocp_solver.set_solution(test)
print(2)
ocp_solver.init_constraints()
print(3)
print("Initial KKT error: ", ocp_solver.KKT_error(t0, q, v))
ocp_solver.solve(t0, q, v)
print("KKT error after convergence: ", ocp_solver.KKT_error(t0, q, v))
print(ocp_solver.get_solver_statistics())
print(4)

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