import robotoc
from tojr import OptJumpTowr
from initialization import calcualate_joint_states,make_sol_feaseble
import numpy as np
import math
import copy


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
joint_efforts_limit = np.full(robot.dimu(), 24.0)
robot.set_joint_effort_limit(joint_efforts_limit)
robot.set_gravity(-3.72)

print("dimu: ", robot.dimu())


dt = 0.05
stand_time = 1.0
t0 = 0.

roll_rand = np.random.rand()*3.14
q_standing = np.array([0., 0., 0.5, 0.0, 0., 0., 1.0, 
                         0.0,  0.2,  .4, -.2, -.4,  #back left
                        -0.0, -0.2,  .4,  .2, -.4, #back right
                        -0.0,  0.2, -.4, -.2,  .4, #front left
                         0.0,  0.2, -.4,  .2,  .4]) #front right
                              ###inner###  ###outer###

# Create the contact sequence
contact_sequence = robotoc.ContactSequence(robot)
mu = 1.0
friction_coefficients = {'FrontLeft_paw': mu, 'BackLeft_paw': mu, 'FrontRight_paw': mu, 'BackRight_paw': mu} 

robot.update_kinematics(q_standing)
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


theta = q_standing[7:].copy()
q = q_standing.copy()
pose = q_standing[:7].copy()
theta = calcualate_joint_states(robot, contact_status_standing,pose, theta)
q[:7] = pose
q[7:] = theta
q_standing = q.copy()

cost = robotoc.CostFunction()
q_ref = q_standing.copy()
q_weight = np.array([1.0, 1.0, 1000.0, 1.0, 1.0, 1.0, 
                        10.0, .0, 0.0, .0, 0.0, 
                        10.0, .0, 0.0, .0, 0.0,
                        10.0, .0, 0.0, .0, 0.0,
                        10.0, .0, 0.0, .0, 0.0])
v_weight = np.full(robot.dimv(), 1.0e-3)
a_weight = np.full(robot.dimv(), 1.0e-3)
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
#constraints.add("joint_position_lower", joint_position_lower)
#constraints.add("joint_position_upper", joint_position_upper)
#constraints.add("joint_velocity_lower", joint_velocity_lower)
#constraints.add("joint_velocity_upper", joint_velocity_upper)
constraints.add("joint_torques_lower", joint_torques_lower)
constraints.add("joint_torques_upper", joint_torques_upper)
constraints.add("friction_cone", friction_cone)



T = t0 + stand_time
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
q=q_standing
v = np.zeros(robot.dimv())
t=t0

ocp_solver.set_solution("q",q_standing)
#f_init = np.array([0,0,0.25])*robot.total_weight()
#ocp_solver.set_solution("f",f_init)
#ocp_solver.set_solution("v",v)
ocp_solver.discretize(t)
ocp_solver.init_constraints()
sol = make_sol_feaseble(robot,ocp_solver.get_solution('q'),ocp_solver.get_time_discretization(),contact_sequence)
ocp_solver.set_solution(sol)


print("Initial KKT error: ", ocp_solver.KKT_error(t, q, v))
ocp_solver.solve(t0, q, v)
print("KKT error after convergence: ", ocp_solver.KKT_error(t, q, v))
print(ocp_solver.get_solver_statistics())



Q = ocp_solver.get_solution('q')
q_last = Q[-1]
print("height: ", q_last[2])
# Display results
viewer = robotoc.utils.TrajectoryViewer(model_info=model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=mu)
viewer.display(ocp_solver.get_time_discretization(), 
               ocp_solver.get_solution('q'), 
               ocp_solver.get_solution('f', 'WORLD'))


