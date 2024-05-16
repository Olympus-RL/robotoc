import pinocchio as pin



import math
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
import time

from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer

import robotoc
import numpy as np
import math





urdf_model_path = '/home/bolivar/OLYMPOC/robotoc/examples/anymal/anymal_b_simple_description/urdf/anymal.urdf'
pkg_path = join(dirname(urdf_model_path), '../..')

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, pkg_path, pin.JointModelFreeFlyer())


## robotoc for visualizing the robot ###
robotoc_model_info = robotoc.RobotModelInfo()
robotoc_model_info.urdf_path = urdf_model_path
robotoc_model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
baumgarte_time_step = 0.05
robotoc_model_info.point_contacts = [robotoc.ContactModelInfo('LF_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('LH_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('RF_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('RH_FOOT', baumgarte_time_step)]

viewer = robotoc.utils.TrajectoryViewer(model_info=robotoc_model_info, viewer_type='gepetto')
viewer.set_contact_info(mu=0.7)




q_standing = np.array([0., 0., 0.4792, 0., 0., 0., 1.0, 
                       -0.1,  0.7, -1.0, 
                       -0.1, -0.7,  1.0, 
                        0.1,  0.7, -1.0, 
                        0.1, -0.7,  1.0])
q_ref = q_standing.copy()
q0 = q_standing

paw_names = [f'{side}{tip}_FOOT' for tip in ['F', 'H'] for side in ['L', 'R']]
print(paw_names)
frame_ids = [ model.getFrameId(paw_name) for paw_name in paw_names]
print(frame_ids)

v0 = np.zeros((model.nv))
v_ref = v0.copy()
data_sim = model.createData()
data_control = model.createData()

contact_models = []
contact_datas = []

for frame_id in frame_ids:
    frame = model.frames[frame_id]
    contact_model = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,frame.parentJoint,frame.placement)

    contact_models.append(contact_model)
    contact_datas.append(contact_model.createData())

pin.initConstraintDynamics(model,data_sim,contact_models)

t = 0
dt = 5e-3


q = q0.copy()
v = v0.copy()
tau = np.zeros((model.nv))

T = 60
dh = 0

dts = []
qs = []

while t <= T:
    if dh > 1:
        print("t:",t)
        dh = 0
    t+=dt

    qs.append(q0)
    dts.append(dt)
    

 
viewer.display(dts, 
               qs,
               None)
    #input()
    

