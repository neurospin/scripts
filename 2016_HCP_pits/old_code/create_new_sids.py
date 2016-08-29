

import json, os

"""
s_ids = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101410', '101915', '102008', "102311", '102513', '102816', '103111', '108121', '103414', '103515', '103818', '104416', '104820', '105115', '105216', '105620', '106016', '106319', '106521', '107018', '107220', '107321', '107422', '107725', '108222', '108323', '108525', '108828']
output_dir = '/home/yl247234/s_ids_lists/'
output = output_dir +'s_ids_mistake.json'
# Save the list of subjects
encoded = json.dumps(s_ids)
with open(output, 'w') as f:
    json.dump(encoded, f)
"""

s_ids = []
path = '/media/yl247234/SAMSUNG/imagen/BVdatabase/'
centres = ['Berlin','Dresden', 'Dublin', 'Hamburg', 'London', 'Mannheim', 'Nottingham', 'Paris']
#centres = ["Berlin"]
for centre in centres:
    path_c = path+centre+'/'
    for s_id in os.listdir(path_c):
        path_s =  path_c+s_id+'/'
        if os.path.isdir(path_s):
            path_file = path_s+'t1mri/BL/default_analysis/segmentation/mesh/'+s_id+'_Lwhite.gii'
            if os.path.isfile(path_file):
                s_ids.append(centre+'/'+s_id)

output_dir = '/home/yl247234/s_ids_lists/'
output = output_dir +'s_ids_imagen_BL.json'
# Save the list of subjects
encoded = json.dumps(s_ids)
with open(output, 'w') as f:
    json.dump(encoded, f)

s_ids = []
path = '/media/yl247234/SAMSUNG/imagen/Freesurfer_mesh_database/imagen/'

for s_id in os.listdir(path):
    s_ids.append(s_id)

output_dir = '/home/yl247234/s_ids_lists/'
output = output_dir +'s_ids_imagen_freesurfer_BL.json'
# Save the list of subjects
encoded = json.dumps(s_ids)
with open(output, 'w') as f:
    json.dump(encoded, f)
