import robotoc
import json

def solution_to_blender_json(robot,solution,td,contact_sequence,save_path):
    frame_names = ["Body"]
    for tail in ["Front","Back"]:
        for side in ["Left","Right"]:
            quad = tail+side
            frame_name = quad + "_motorHousing"
            frame_names.append(frame_name)
            for a in ["inner","outer"]:
                for b in ["hip","shank"]:
                    frame_name = "_".join([quad,b,a,"visual"])
                    frame_names.append(frame_name)

    frame2index = {}



    log = [] 
   
    for i in range(len(td)):
        grid = td[i]
        s = solution[i]
        phase = grid.phase
        contact_status = contact_sequence.contact_status(phase)
        robot.update_kinematics(solution[i].q)
        
        F = []
        P =[]
        for j in range(robot.max_num_contacts()):
            if contact_status.is_contact_active(j):
                f_local = solution[i].f_contact[j]
                contact_pos = robot.frame_position(contact_status.contact_frame_name(j))
                f_world = robot.frame_rotation(contact_status.contact_frame_name(j)) @ f_local[:3]
                F.append(f_world)
                P.append(contact_pos)
            else:
                F.append(None)
                P.append(None)
        q = solution[i].q
        t = grid.t

        frame_log = {
            "t": t,
            "q": q.tolist(),
            "F": [f.tolist() if f is not None else None for f in F],
            "P": [p.tolist() if p is not None else None for p in P]
        }

        log.append(frame_log)

    with open(save_path, 'w') as outfile:
        json.dump(log, outfile)
    
        
 

