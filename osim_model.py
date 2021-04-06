"""
Functions to modify muscles and joints of osim model
"""

def modify_Millard(osim_file, ignore_tendon=True, fiber_damping=0.001, ignore_dyn=False):
    """
    Modify Millard muscles parameters
    INPUTS: - osim_file: string, path to osim model
            - ignore_tendon: bool, to ignore or not tendon compliance
            - fiber_damping: float, muscle fiber damping
            - ignore_dyn: bool, to ignore or not muscle excitation to activation dynamics
    OUTPUT: - osim_file: string, path to modified osim model
    """
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if line.split()[0].split('>')[0] == '<ignore_tendon_compliance':
            new_lines[l] = '                    <ignore_tendon_compliance>'+ignore_tendon+'</ignore_tendon_compliance>\n'
        elif line.split()[0].split('>')[0] == '<fiber_damping':
            new_lines[l] = '                    <fiber_damping>'+str(fiber_damping)+'</fiber_damping>\n'
        elif line.split()[0].split('>')[0] == '<ignore_activation_dynamics':
            new_lines[l] = '                    <ignore_activation_dynamics>'+ignore_dyn+'</ignore_activation_dynamics>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def able_Muscle(osim_file, muscles, able):
    """
    Able or disable some muscles
    INPUTS: - osim_file: string, path to osim model
            - muscles: string array, list of muscles to able or disable
            - able: bool, to able or disable the previous muscles
    OUTPUT: - osim_file: string, path to modified osim model
    """
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Millard2012EquilibriumMuscle':
            if line.split()[1].split('"')[1] in muscles:
                new_lines[l+2] = '                    <appliesForce>'+able+'</appliesForce>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def lock_Coord(osim_file, coords, lock):
    """
    Lock or unlock some coordinates
    INPUTS: - osim_file: string, path to osim model
            - coords: string array, list of coordinates to lock or unlock
            - lock: bool, to lock or unlock the previous coordinates
    OUTPUT: - osim_file: string, path to modified osim model
    """
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
            if line.split()[1].split('"')[1] in coords:
                new_lines[l+10] = '                            <locked>'+lock+'</locked>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file


def modify_default_Coord(osim_file, coord, value):
    """
    Modify the default value of a coordinate
    INPUTS: - osim_file: string, path to osim model
            - coord: string, coordinate to modify
            - value: float, new default value
    OUTPUT: - osim_file: string, path to modified osim model
    """
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    for l in range(len(lines)):
        line = lines[l]
        if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
            if line.split()[1].split('"')[1] == coord:
                new_lines[l + 2] = '                            <default_value>'+str(value)+'</default_value>\n'
    with open(osim_file, 'w') as file:
        file.writelines(new_lines)
    return osim_file
