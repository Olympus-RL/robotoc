import xml.etree.ElementTree as ET

# Define the scalar value
c = 10  # You can change this value as needed
print(f'Scaling all mass and inertia properties by {c}')

# Load the URDF file
tree = ET.parse('olympus_open.urdf')
root = tree.getroot()

# Define a function to scale mass and inertia properties
def scale_mass_inertia(mass_elem, inertia_elem):
    mass_value = float(mass_elem.attrib['value']) * c
    inertia_values = [float(inertia_elem.attrib[attr]) * c for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']]
    
    mass_elem.attrib['value'] = str(mass_value)
    for i, attr in enumerate(['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']):
        inertia_elem.attrib[attr] = str(inertia_values[i])

# Find all links and scale their mass and inertia properties
for link in root.findall('link'):
    if 'Body' in link.attrib['name'] or 'motorHousing' in link.attrib['name']:
        print(f'Skipping {link.attrib["name"]}')
        continue
    inertial = link.find('inertial')
    if inertial is not None:
        mass = inertial.find('mass')
        inertia = inertial.find('inertia')
        if mass is not None and inertia is not None:
            scale_mass_inertia(mass, inertia)

# Save the modified URDF file
tree.write(f'olympus_open_scaled.urdf')
