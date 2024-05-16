import robotoc
import pinocchio as pin

import math
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
import time


pkg_dir = '/home/bolivar/OLYMPOC/robotoc/descriptions'
urdf_model_path = pkg_dir + '/olympus_description/urdf/olympus.urdf'
mesh_dir = '/home/bolivar/OLYMPOC/robotoc/descriptions/olympus/meshes/'
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path,pkg_dir, root_joint=pin.JointModelFreeFlyer())



q0 = np.zeros(model.nq)
q0[2] = 0.52
q0[6] = 1
q_ref = q0.copy() + 0.5


paw_names = [f'{tip}{side}_paw' for tip in ['Front', 'Back'] for side in ['Left', 'Right']]
paw_frame_ids = [ model.getFrameId(paw_name) for paw_name in paw_names]
inner_shank_names = [f'{tip}{side}_shank_inner' for tip in ['Front', 'Back'] for side in ['Left', 'Right']]
outer_shank_names = [f'{tip}{side}_shank_outer' for tip in ['Front', 'Back'] for side in ['Left', 'Right']]

v0 = np.zeros((model.nv))
v_ref = v0.copy()
data_sim = model.createData()
data_control = model.createData()

contact_models = []
contact_datas = []


for frame_id in paw_frame_ids:
    frame = model.frames[frame_id]
    contact_model = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,frame.parentJoint,frame.placement)
    contact_models.append(contact_model)
    contact_datas.append(contact_model.createData())

for inner_id, outer_id in zip(inner_shank_names, outer_shank_names):
    inner_frame_id = model.getFrameId(inner_id)
    outer_frame_id = model.getFrameId(outer_id)
    inner_frame = model.frames[inner_frame_id]
    outer_frame = model.frames[outer_frame_id]
    x = np.array([0,0,-0.296])
    ancle_joint_pos = pin.SE3().Identity()
    ancle_joint_pos.translation = x
    print(ancle_joint_pos.rotation)
    contact_model = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,inner_frame.parentJoint,ancle_joint_pos,outer_frame.parentJoint,ancle_joint_pos)
    contact_models.append(contact_model)
    contact_datas.append(contact_model.createData())

num_constraints = len(paw_frame_ids) + len(inner_shank_names)
contact_dim = 3 * num_constraints

pin.initConstraintDynamics(model,data_sim,contact_models)

t = 0
dt = 5e-3

S = np.zeros((model.nv-6,model.nv))
S.T[6:,:] = np.eye(model.nv-6)

Kp_posture = 10.
Kv_posture = 0.05*math.sqrt(Kp_posture)

q = q0.copy()
v = v0.copy()
tau = np.zeros((model.nv))

T = 10
dts = []
q_traj = []
time_since_last_print = 0
render_dt = 0.1
time_since_last_render = 0

while t <= T:
   
    if time_since_last_print > 1:
        time_since_last_print = 0
        print("t:",t)

    if time_since_last_render > render_dt:
        time_since_last_render = 0
        q_traj.append(q.copy())
        dts.append(render_dt)
    t += dt
    time_since_last_print += dt
    time_since_last_render += dt

    

    tic = time.time()
    J_constraint = np.zeros((contact_dim,model.nv))
    pin.computeJointJacobians(model,data_control,q)
    constraint_index = 0
    for k in range(num_constraints):
        contact_model = contact_models[k]
        J_constraint[constraint_index:constraint_index+3,:] = pin.getFrameJacobian(model,data_control,contact_model.joint1_id,contact_model.joint1_placement,contact_model.reference_frame)[:3,:]
        constraint_index += 3
    A = np.vstack((S,J_constraint))
    b = pin.rnea(model,data_control,q,v,np.zeros((model.nv)))

    sol = np.linalg.lstsq(A.T,b,rcond=None)[0]
    tau = np.concatenate((np.zeros((6)),sol[:model.nv-6]))

    tau[6:] += -Kp_posture*(pin.difference(model,q_ref,q))[6:] - Kv_posture*(v - v_ref)[6:]
   

    prox_settings = pin.ProximalSettings(1e-12,1e-12,10)
    a = pin.constraintDynamics(model,data_sim,q,v,tau,contact_models,contact_datas,prox_settings)
    #print("a:",a.T)
    #print("v:",v.T)
    #print("constraint:",np.linalg.norm(J_constraint@a))
    #print("iter:",prox_settings.iter)

    v += a * dt
    q = pin.integrate(model,q,v*dt)
   

## robotoc for visualizing the robot ###
import robotoc
robotoc_model_info = robotoc.RobotModelInfo()
robotoc_model_info.urdf_path = urdf_model_path
robotoc_model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
baumgarte_time_step = 0.05
robotoc_model_info.point_contacts = [robotoc.ContactModelInfo(paw_name, baumgarte_time_step) for paw_name in paw_names]

viewer = robotoc.utils.TrajectoryViewer(model_info=robotoc_model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=0.7)

print("displaying")
print("len_q_traj:",len(q_traj))
print("len_dts:",len(dts))

viewer.display(dts,q_traj,None)
