import robotoc
from tojr import OptJumpTowr
from calculate_joint_states import calcualate_joint_states
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
model_info.ckcs = [robotoc.CKCInfo("FrontLeft_ankle_outer","FrontLeft_ankle_inner",baumgarte_time_step),
                    robotoc.CKCInfo("FrontRight_ankle_outer","FrontRight_ankle_inner",baumgarte_time_step),
                    robotoc.CKCInfo("BackLeft_ankle_outer","BackLeft_ankle_inner",baumgarte_time_step),
                    robotoc.CKCInfo("BackRight_ankle_outer","BackRight_ankle_inner",baumgarte_time_step)]

robot = robotoc.Robot(model_info)
robot.set_gravity(-3.72)



dt = 0.01
jump_length = np.array([3.5, 0, 0])
take_off_duration = 1.0
flight_duration = 0.20
touch_down_duration = 0.5
t0 = 0.

q_standing = np.array([0., 0., 0.5, np.sqrt(2)/2, 0., 0., np.sqrt(2)/2, 
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

print("contact_positions_standing: ", contact_positions_standing)

contact_status_standing = robot.create_contact_status()
contact_status_standing.activate_contacts(['FrontLeft_paw', 'BackLeft_paw', 'FrontRight_paw', 'BackRight_paw'])
contact_status_standing.set_contact_placements(contact_positions_standing)
contact_status_standing.set_friction_coefficients(friction_coefficients)
contact_sequence.init(contact_status_standing)

contact_status_flying = robot.create_contact_status()
contact_sequence.push_back(contact_status_flying, t0+take_off_duration, sto=False)


T = t0 + take_off_duration + flight_duration + touch_down_duration
N = math.floor(T/dt) 
td = robotoc.TimeDiscretization(T,N,0)
td.discretize(contact_sequence,t0)
td.correct_time_steps(contact_sequence,t0)

Q = []
theta_guess = np.array([ 0.0,  1.0,  1.0, -1.0, -1.4,  #back left
                        -0.0, -1.0,  1.0, 1.0, -1.4, #back right
                        -0.0,  1.0, -1.0, -1.0, 1.4, #front left
                         0.0,  1.0, -1.0, 1.0, 1.4]) #front right
theta = theta_guess
dh = (q_standing[2]-0.20)/len(td)
q = q_standing.copy()
h = q_standing[1]
for i in range(len(td)):
    pose = q_standing[:7].copy()
    pose[1] = -h
    theta = calcualate_joint_states(robot, contact_positions_standing,pose, theta)
    q[:7] = pose
    q[7:] = theta
    Q.append(q.copy())
    #sQ.append(q_standing)
    h -= dh
    #print(h)


# Display results
viewer = robotoc.utils.TrajectoryViewer(model_info=model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=mu)
viewer.display(td, Q)



