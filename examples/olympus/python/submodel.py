
import pinocchio as pin
import numpy as np
 
from os.path import *
 
# Goal: Build a reduced model from an existing URDF model by fixing the desired joints at a specified position.
 
# Load UR 

urdf_filename = '/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus.urdf'
mesh_dir = '/home/bolivar/OLYMPOC/robotoc/descriptions/'
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_filename, mesh_dir,pin.JointModelFreeFlyer())
 
# Check dimensions of the original model
print('standard model: dim=' + str(len(model.joints)))
for jn in model.joints:
    print(jn)
print('-' * 30)


joints_to_keep_ID = [3,4,5,6]
joints_to_lock_ID = [i for i in range(1,len(model.joints)+1) if i not in joints_to_keep_ID]
 

 
initialJointConfig = np.zeros(model.nq)
initialJointConfig[6] = 1.0
 
# Option 1: Only build the reduced model in case no display needed:
model_reduced = pin.buildReducedModel(model, joints_to_lock_ID, initialJointConfig)
reduced_model_data = model_reduced.createData()

print('reduced model: dim=' + str(len(model_reduced.joints)))
for name in model_reduced.names:
    print(name)
print('-' * 30)
