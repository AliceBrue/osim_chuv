import numpy as np
import xlrd
import matplotlib.pyplot as plt
import os
from control_SO import *


def main():
    """
    Script to evaluate kinematics to EMG data:
    Extract kinematics data of 'movement' of interest from 'kinematics.xls'
    Movement data are scaled from subject measures during 'static' period, and saved in osim format .trc in '
        kin_folder/movement/' folder
    Joint angles are also computed and saved in the previous folder
    If an osim IK_file computed from previous kinematics data is provided (in kin_folder/movement/ folder),
        IK and previous joint angles are compared
    If osim cmc_files computed from previous IK_file is provided (in kin_folder/movement/ folder), CMC muscle
        activation and recorded EMG are compared
    """
    # Movement of interest
    kin_folder = 'kinematics/'
    emg_folder = 'emg/'
    model_name = 'Henri'
    movement = 'sh_flexion'  # 'sh_flexion', 'elb_flexion', 'arm_flexion' or 'sh_adduction'
    IK_file = None  # None or path (kin_folder+movement+'/joints_IK_'+movement+'.mot')
    cmc_files = None  # None or path (kin_folder+movement+'/'+model_name+'_')
    cmc_ti = 1.48
    cmc_tf = 2.0
    delay_video = 108.4 - 55.48

    # Static kin data for scaling
    if movement in ['sh_flexion', 'elb_flexion', 'arm_flexion']:
        ti = 1.24
        tf = 1.30
        markers = ['shoulder', 'elbow', 'wrist']
        scaling_ref = {'shoulder_hip': 0.55, 'shoulder_elbow': 0.34, 'elbow_wrist': 0.28, 'shoulder_ear': 0.22,
                       'ear_eye': 0.13}
        ground_ref = ['shoulder', 'shoulder', 0.2]
        static_scaling, static_ground_ref = scaling_kin_data(kin_folder, 'static_'+movement, ti, tf, markers,
                                                             scaling_ref, ground_ref, conf_th=0.7, plot=False)
    elif movement in ['sh_adduction']:
        ti = 2.50
        tf = 2.51
        markers = ['shoulder', 'elbow', 'wrist']
        scaling_ref = {'shoulder_shoulder': 0.42, 'shoulder_hip': 0.55, 'shoulder_elbow': 0.34, 'elbow_wrist': 0.28}
        ground_ref = [0.0, 'shoulder', 'shoulder']
        static_scaling, static_ground_ref = scaling_kin_data(kin_folder, 'static_'+movement, ti, tf, markers,
                                                             scaling_ref, ground_ref, conf_th=0.7, plot=False)

    # Movement kin data
    if movement == 'sh_flexion':
        ti = 1.48
        tf = 2.0
        joint_angles = scaled_kin_data(kin_folder, movement, ti, tf, markers, static_scaling, static_ground_ref,
                                       plot=False)
        if IK_file is not None:
            comp_IK_joints(IK_file, joint_angles, ['elv_angle', 'shoulder_rot'])

    elif movement == 'elb_flexion':
        ti = 1.24
        tf = 1.47
        joint_angles = scaled_kin_data(kin_folder, movement, ti, tf, markers, static_scaling, static_ground_ref,
                                       plot=False)
        if IK_file is not None:
            comp_IK_joints(IK_file, joint_angles, ['elv_angle', 'shoulder_rot'])

    elif movement == 'arm_flexion':
        ti = 2.18
        tf = 2.29
        joint_angles = scaled_kin_data(kin_folder, movement, ti, tf, markers, static_scaling, static_ground_ref,
                                       plot=False)
        if IK_file is not None:
            comp_IK_joints(IK_file, joint_angles, ['elv_angle', 'shoulder_rot'])

    elif movement == 'sh_adduction':
        ti = 2.50
        tf = 3.02
        joint_angles = scaled_kin_data(kin_folder, movement, ti, tf, markers, static_scaling, static_ground_ref,
                                       plot=False)
        if IK_file is not None:
            comp_IK_joints(IK_file, joint_angles, ['elv_angle', 'shoulder_rot'])

    # CMC results
    if cmc_files is not None:
        cmc_states_file = cmc_files+'states.sto'
        cmc_force_file = cmc_files+'Actuation_force.sto'
        cmc_error_file = cmc_files+'pErr.sto'
        plot_cmc_reserve(cmc_force_file, cmc_error_file, 4)
        comp_cmc_emg(cmc_states_file, emg_folder, cmc_ti, cmc_tf, delay_video,
                     disabled_muscles=['TRAPM', 'RHS', 'RHI', 'LAT1', 'LAT2', 'LAT3', 'SUP'], cmc_exc_file=None,
                     n_emg=8, emg_step=0.5)

    plt.show()


def scaling_kin_data(kin_folder, movement, ti, tf, markers, scaling_ref, ground_ref, time_step=0.0333, conf_th=0.5,
                     plot=False):
    """
    Extract kinematics data of 'static' period of interest from 'kinematics.xls'
    Compute a scaling factor from subject measures (print openpifpaf confidences and scaling std)
    Save extracted data in osim format .trc in 'kin_folder/movement/' folder
    INPUTS: - kin_folder: string, path of the folder where 'kinematics.xls' is and extracted data will be saved
            - movement: string, name of the static period of interest in ['static_sh_flexion', 'static_sh_adduction']
            - ti: float, static period initial time in video
            - tf: float, static period final time in video
            - markers: string array, markers kin data to save in osim format .trc
            - scaling_ref: dictionary, subject measures ({'shoulder_hip': 0.55, ...}}
            - ground_ref: array, osim ground references
            - time_step: float, openpifpaf time step
            - conf_th: float, confidence threshold above which openpifpaf marker are rejected
            - plot: bool, to plot or not the extracted data
    OUTPUTS: - scaling: float, computed scaling factor
             - static_ground_ref: array, osim ground references
    """
    # kin data
    workbook = xlrd.open_workbook(kin_folder+'kinematics.xls')
    sheet = workbook.sheet_by_index(0)
    bodies = []
    for i in range(sheet.ncols):
        bodies.append(sheet.cell_value(0, i))
    scale_bodies = []
    for ref in list(scaling_ref.keys()):
        ref1 = ref.split('_')[0]
        ref2 = ref.split('_')[1]
        if ref1 == ref2:
            scale_bodies.append('left_'+ref1)
        if ref1 not in markers and ref1 not in scale_bodies:
            scale_bodies.append(ref1)
        if ref2 not in markers and ref2 not in scale_bodies:
            scale_bodies.append(ref2)
    markers_i = np.zeros(len(markers))
    scale_bodies_i = np.zeros(len(scale_bodies))
    for i in range(len(markers)):
        markers_i[i] = bodies.index('right_'+markers[i]+'_x')
    markers_i = markers_i.astype(int)
    for i in range(len(scale_bodies)):
        if scale_bodies[i].split('_')[0] == 'left':
            scale_bodies_i[i] = bodies.index(scale_bodies[i] + '_x')
        else:
            scale_bodies_i[i] = bodies.index('right_'+scale_bodies[i]+'_x')
    scale_bodies_i = scale_bodies_i.astype(int)

    period = [ti // 1 * 60 + ti % 1 * 100, tf // 1 * 60 + tf % 1 * 100]
    time = np.zeros(int((period[1]-period[0])/time_step))
    kin_data = np.zeros((int((period[1]-period[0])/time_step), len(markers)*3))
    scale_data = np.zeros((int((period[1]-period[0])/time_step), len(scale_bodies)*2))
    confidence_data = np.zeros((int((period[1]-period[0])/time_step), len(markers)+len(scale_bodies)))

    for t in range(1, sheet.nrows):
        if float(sheet.cell_value(t, 0)) < period[0] and float(sheet.cell_value(t+1, 0)) > period[0]:
            t0 = t+1
        if float(sheet.cell_value(t, 0)) > period[0] and float(sheet.cell_value(t, 0)) < period[1]:
            time[t-t0] = sheet.cell_value(t, 0)-sheet.cell_value(t0, 0)
            for i in range(len(markers)):
                kin_data[t-t0, 3*i] = sheet.cell_value(t, markers_i[i])
                kin_data[t - t0, 3*i+1] = sheet.cell_value(t, markers_i[i]+1)
                confidence_data[t - t0, i] = sheet.cell_value(t, markers_i[i] + 2)
            for i in range(len(scale_bodies)):
                scale_data[t-t0, 2*i] = sheet.cell_value(t, scale_bodies_i[i])
                scale_data[t - t0, 2*i+1] = sheet.cell_value(t, scale_bodies_i[i]+1)
                confidence_data[t - t0, len(markers)+i] = sheet.cell_value(t, scale_bodies_i[i]+2)
    # confidence
    confidence_ref = np.mean(confidence_data, axis=0)
    print('confidences:', markers, scale_bodies)
    print('\t', confidence_ref)
    for i in range(len(confidence_ref)):
        if confidence_ref[i] < conf_th:
            if i < len(markers):
                unconf_body = markers[i]
            else:
                unconf_body = scale_bodies[i-len(markers)]
            for ref in list(scaling_ref.keys()):
                if unconf_body in ref.split('_'):
                    scaling_ref.pop(ref)

    # scaling
    l_ref = np.zeros(len(scaling_ref))
    std_ref = np.zeros(len(scaling_ref))
    for i in range(len(list(scaling_ref.keys()))):
        ref = list(scaling_ref.keys())[i]
        ref1 = ref.split('_')[0]
        ref2 = ref.split('_')[1]
        if ref1 == ref2:
            l = np.sqrt((kin_data[:, 3 * int(markers.index(ref1))] - scale_data[:, 3 * int(scale_bodies.index('left_'+ref2))]) ** 2 +
                        (kin_data[:, 3 * int(markers.index(ref1)) + 1] - scale_data[:,
                                                                         3 * int(scale_bodies.index('left_'+ref2)) + 1]) ** 2)
            l_ref[i] = np.mean(l)
            std_ref[i] = np.std(l)
        else:
            if ref1 in markers:
                if ref2 in markers:
                    l = np.sqrt((kin_data[:, 3*int(markers.index(ref1))] - kin_data[:, 3*int(markers.index(ref2))])**2 +
                                    (kin_data[:, 3*int(markers.index(ref1))+1] - kin_data[:, 3*int(markers.index(ref2))+1])**2)
                    l_ref[i] = np.mean(l)
                    std_ref[i] = np.std(l)
                else:
                    l = np.sqrt((kin_data[:, 3*int(markers.index(ref1))] - scale_data[:, 2*int(scale_bodies.index(ref2))])**2 +
                                    (kin_data[:, 3*int(markers.index(ref1))+1] - scale_data[:, 2*int(scale_bodies.index(ref2))+1])**2)
                    l_ref[i] = np.mean(l)
                    std_ref[i] = np.std(l)
            else:
                if ref2 in markers:
                    l = np.sqrt((kin_data[:, 3*int(markers.index(ref2))] - scale_data[:, 2*int(scale_bodies.index(ref1))])**2 +
                        (kin_data[:, 3*int(markers.index(ref2))+1] - scale_data[:, 2*int(scale_bodies.index(ref1))+1])**2)
                    l_ref[i] = np.mean(l)
                    std_ref[i] = np.std(l)
                else:
                    l = np.sqrt((scale_data[:, 2*int(scale_bodies.index(ref1))] -
                                     scale_data[:, 2*int(scale_bodies.index(ref2))])**2 +
                                    (scale_data[:, 2*int(scale_bodies.index(ref1))+1] -
                                     scale_data[:, 2*int(scale_bodies.index(ref2))+1])**2)
                    l_ref[i] = np.mean(l)
                    std_ref[i] = np.std(l)
    print('std ref:', list(scaling_ref.keys()))
    print('\t', std_ref)
    scaling = np.divide(l_ref, list(scaling_ref.values()))
    print('scaling:', scaling, ', mean:', np.mean(scaling), ', std:', np.std(scaling))
    scaling = np.mean(scaling)

    if movement.split('_')[0] == 'static':
        if ground_ref[0] in markers and ground_ref[1] in markers and isinstance(ground_ref[2], float):
                static_ground_ref = [np.mean(kin_data[:, 3 * int(markers.index(ground_ref[0]))]),
                                     np.mean(kin_data[:, 3 * int(markers.index(ground_ref[1]))+1]), ground_ref[2]]
        elif ground_ref[1] in markers and ground_ref[2] in markers and isinstance(ground_ref[0], float):
                static_ground_ref = [ground_ref[0], np.mean(kin_data[:, 3 * int(markers.index(ground_ref[1]))+1]),
                                     np.mean(kin_data[:, 3 * int(markers.index(ground_ref[2]))])]
        else:
            print('TO DO')
        for i in range(len(markers)):
            kin_data[:, 3*i] = (kin_data[:, 3*i] - static_ground_ref[0])/scaling
            kin_data[:, 3*i+1] = (static_ground_ref[1] - kin_data[:, 3*i+1])/scaling + 1
            kin_data[:, 3*i+2] = static_ground_ref[2] * np.ones(len(kin_data[:, 0]))

    # plot kin data
    if plot:
        plt.figure()
        plt.plot(kin_data[:,0], label='sx')
        plt.plot(kin_data[:,1], label='sy')
        plt.plot(kin_data[:,3], label='ex')
        plt.plot(kin_data[:,4], label='ey')
        plt.plot(kin_data[:,6], label='wx')
        plt.plot(kin_data[:,7], label='wy')
        plt.legend()
        plt.show()

    # write kin file
    kin_file = open(kin_folder+'kin_'+movement+'.trc', 'w')
    kin_file.write('PathFileType\t4\t(X/Y/Z)\t'+movement+'.trc\n')
    kin_file.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    kin_file.write('30.00\t30.00\t'+str(len(time))+'\t3\tmm\t30.00\t1\t'+str(len(time))+'\n')
    kin_file.write('Frame#\tTime\t')
    for m in markers:
        kin_file.write(m+'\t\t\t')
    kin_file.write('\n')
    kin_file.write('\t\t')
    for i in range(len(markers)):
        kin_file.write('X'+str(i+1)+'\tY'+str(i+1)+'\tZ'+str(i+1)+'\t')
    kin_file.write('\n')
    kin_file.write('\n')
    frame = 1
    for t in range(len(time)):
        line = str(frame)+'\t'+str(time[t])
        for i in range(3*len(markers)):
            line += '\t'+str(kin_data[t, i]*1000)
        line += '\n'
        kin_file.write(line)
        frame += 1

    return scaling, static_ground_ref


def scaled_kin_data(kin_folder, movement, ti, tf, markers, scaling, ground_ref, time_step=0.0333,
                    joints=['shoulder_elev', 'elbow_flexion'], plot=False):
    """
    Extract kinematics data of 'movement 'of interest from 'kinematics.xls'
    Scale the data from previous 'static' scaling factor and compute joint angles of interest (print openpifpaf
        confidences)
    Save extracted data in osim format .trc and oint angles in osim format '.mot' in 'kin_folder/movement/' folder
    INPUTS: - kin_folder: string, path of the folder where 'kinematics.xls' is and extracted data will be saved
            - movement: string, name of the movement of interest in ['sh_flexion', 'elb_flexion', 'arm_flexion',
                         'sh_adduction']
            - ti: float, movement initial time in video
            - tf: float, movement final time in video
            - markers: string array, markers kin data to save in osim format .trc
            - scaling: static' scaling factor
            - ground_ref: array, osim ground references
            - time_step: float, openpifpaf time step
            - joints: string array, joint angles to compute in ['shoulder_elev', 'elbow_flexion']
            - plot: bool, to plot or not joint angles
    OUTPUTS: - joint_angles: dictionary, computed joint angles ({'time': float array, 'shoulder_elev': float array...})
    """
    # kin data
    workbook = xlrd.open_workbook(kin_folder+'kinematics.xls')
    sheet = workbook.sheet_by_index(0)
    bodies = []
    for i in range(sheet.ncols):
        bodies.append(sheet.cell_value(0, i))
    markers_i = np.zeros(len(markers))
    for i in range(len(markers)):
        markers_i[i] = bodies.index('right_'+markers[i]+'_x')
    markers_i = markers_i.astype(int)

    period = [ti // 1 * 60 + ti % 1 * 100, tf // 1 * 60 + tf % 1 * 100]
    time = np.zeros(int((period[1]-period[0])/time_step))
    kin_data = np.zeros((int((period[1]-period[0])/time_step), len(markers)*3))
    confidence_data = np.zeros((int((period[1]-period[0])/time_step), len(markers)))

    for t in range(1, sheet.nrows):
        if float(sheet.cell_value(t, 0)) < period[0] and float(sheet.cell_value(t+1, 0)) > period[0]:
            t0 = t+1
        if float(sheet.cell_value(t, 0)) > period[0] and float(sheet.cell_value(t, 0)) < period[1]:
            time[t-t0] = sheet.cell_value(t, 0)-sheet.cell_value(t0, 0)
            for i in range(len(markers)):
                kin_data[t-t0, 3*i] = sheet.cell_value(t, markers_i[i])
                kin_data[t - t0, 3*i+1] = sheet.cell_value(t, markers_i[i]+1)
                confidence_data[t - t0, i] = sheet.cell_value(t, markers_i[i] + 2)

    # confidence
    confidence_ref = np.mean(confidence_data, axis=0)
    print('confidences:', markers)
    print('\t', confidence_ref)

    # joint angles
    joint_angles = {}
    for j in joints:
        if j == 'shoulder_elev':
            y_humerus = kin_data[:, 4] - kin_data[:, 1]
            x_humerus = kin_data[:, 3] - kin_data[:, 0]
            l_humerus = np.sqrt((kin_data[:, 3] - kin_data[:,0])**2+(kin_data[:,4] - kin_data[:,1])**2)
            l_humerus = np.mean(l_humerus)
            joint_angles[j] = np.arccos(np.maximum(np.minimum(y_humerus / l_humerus, 1), -1)) * 180 / 3.14
            for t in range(len(y_humerus)):
                if y_humerus[t] / l_humerus > 1:
                    joint_angles[j][t] = (np.arcsin(np.maximum(np.minimum(x_humerus[t] / l_humerus, 1), -1)) * 180 / 3.14)
        elif j == 'elbow_flexion':
            y_radius = kin_data[:, 7] - kin_data[:, 4]
            x_radius = kin_data[:, 6] - kin_data[:, 3]
            l_radius = np.sqrt((kin_data[:, 6] - kin_data[:, 3])**2 + (kin_data[:,7] - kin_data[:, 4])**2)
            l_radius = np.mean(l_radius)
            joint_angles[j] = np.arccos(np.maximum(np.minimum(y_radius/l_radius, 1), -1))*180/3.14 - \
                              joint_angles['shoulder_elev']
            for t in range(len(y_radius)):
                if y_radius[t] / l_radius > 1:
                    joint_angles[j][t] = np.arcsin(np.maximum(np.minimum(x_radius[t]/l_radius, 1), -1))*180/3.14  - \
                              joint_angles['shoulder_elev'][t]
        else:
            print('TO DO')
    joint_angles['time'] = time

    # write joint angles file
    file_path = open(kin_folder+movement+'/joints_'+movement+'.mot', 'w')
    file_path.write('Joints_'+movement+'\nversion=1\nnRows='+str(len(joint_angles['time'])) +
                    '\nnColumns='+str(len(joints))+'\ninDegrees=yes\nendheader\n')
    file_path.write('time')
    for j in joints:
        file_path.write('\t'+j)
    file_path.write('\n')
    for t in range(len(joint_angles['time'])):
        file_path.write(str(joint_angles['time'][t]))
        for j in joints:
            file_path.write('\t' + str(joint_angles[j][t]))
        file_path.write('\n')

    if movement in ['sh_flexion', 'elb_flexion', 'arm_flexion']:
        if not (isinstance(ground_ref[0], float) and isinstance(ground_ref[1], float) and
                isinstance(ground_ref[2], float)):
            print('TO DO')
        for i in range(len(markers)):
            kin_data[:, 3 * i] = (kin_data[:, 3 * i] - ground_ref[0]) / scaling
            kin_data[:, 3 * i + 1] = (ground_ref[1] - kin_data[:, 3 * i + 1]) / scaling + 1
            kin_data[:, 3 * i + 2] = ground_ref[2] * np.ones(len(kin_data[:, 0]))

    elif movement in ['sh_adduction']:
        if not (isinstance(ground_ref[0], float) and isinstance(ground_ref[1], float) and
                isinstance(ground_ref[2], float)):
            print('TO DO')
        for i in range(len(markers)):
            kin_data[:, 3 * i + 1] = (ground_ref[1] - kin_data[:, 3 * i + 1]) / scaling + 1
            kin_data[:, 3 * i + 2] = (kin_data[:, 3 * i] - ground_ref[2]) / scaling + 0.2
            kin_data[:, 3 * i] = ground_ref[0] * np.ones(len(kin_data[:, 0]))

    # plot joint angles
    if plot:
        if movement in ['sh_flexion', 'elb_flexion', 'arm_flexion']:
            plt.figure()
            plt.plot(kin_data[:,0], label='sx')
            plt.plot(kin_data[:,1], label='sy')
            plt.plot(kin_data[:,3], label='ex')
            plt.plot(kin_data[:,4], label='ey')
            plt.plot(kin_data[:,6], label='wx')
            plt.plot(kin_data[:,7], label='wy')
            plt.legend()
            plt.show()
        elif movement in ['sh_adduction']:
            plt.figure()
            plt.plot(kin_data[:,2], label='sz')
            plt.plot(kin_data[:,1], label='sy')
            plt.plot(kin_data[:,5], label='ez')
            plt.plot(kin_data[:,4], label='ey')
            plt.plot(kin_data[:,8], label='wz')
            plt.plot(kin_data[:,7], label='wy')
            plt.legend()
            plt.show()

    # write kin file
    kin_file = open(kin_folder+movement+"/kin_"+movement+".trc", 'w')
    kin_file.write('PathFileType\t4\t(X/Y/Z)\t'+movement+'.trc\n')
    kin_file.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    kin_file.write('30.00\t30.00\t'+str(len(time))+'\t3\tmm\t30.00\t1\t'+str(len(time))+'\n')
    kin_file.write('Frame#\tTime\t')
    for m in markers:
        kin_file.write(m+'\t\t\t')
    kin_file.write('\n')
    kin_file.write('\t\t')
    for i in range(len(markers)):
        kin_file.write('X'+str(i+1)+'\tY'+str(i+1)+'\tZ'+str(i+1)+'\t')
    kin_file.write('\n')
    kin_file.write('\n')
    frame = 1
    for t in range(len(time)):
        line = str(frame)+'\t'+str(time[t])
        for i in range(3*len(markers)):
            line += '\t'+str(kin_data[t, i]*1000)
        line += '\n'
        kin_file.write(line)
        frame += 1

    return joint_angles


def comp_IK_joints(IK_file, joint_angles, add_IK_joints=[]):
    """
    Compare osim IK joint angles with previously computed joint angles from kin data by plotting them and printing the
        MAE
    INPUTS: - IK_file: string, path of osim IK file
            - joint_angles: dictionary, computed joint angles ({'time': float array, 'shoulder_elev': float array...})
            - add_IK_joints: string array, additional IK joint angles to plot
    """
    # IK data
    IK_data = open(IK_file, 'r')
    lines = IK_data.readlines()
    IK_joints = lines[10].split()[1:]
    joints = list(joint_angles.keys())[:-1]
    joints_i = np.zeros(len(joints)+len(add_IK_joints))
    for i in range(len(joints)):
        joints_i[i] = IK_joints.index(joints[i]) + 1
    for i in range(len(add_IK_joints)):
        joints_i[i+len(joints)] = IK_joints.index(add_IK_joints[i]) + 1
    joints_i = joints_i.astype(int)
    IK_angles = np.zeros((len(lines[11:]), len(joints)+len(add_IK_joints)))
    IK_time = np.zeros(len(lines[11:]))
    for l in range(len(lines[11:])):
        IK_time[l] = lines[l+11].split()[0]
        for i in range(len(joints)):
            IK_angles[l, i] = lines[l+11].split()[joints_i[i]]
        for i in range(len(add_IK_joints)):
            IK_angles[l, i+len(joints)] = lines[l+11].split()[joints_i[i+len(joints)]]

    # MAE
    for i in range(len(joints)):
        print(joints[i]+ ' MAE:', np.mean(abs(IK_angles[:, i]-joint_angles[joints[i]])))

    # plot
    plt.figure()
    for i in range(len(joints)):
        plt.subplot(len(joints), 1, i+1)
        plt.plot(joint_angles['time'], joint_angles[joints[i]], label=joints[i])
        plt.plot(IK_time, IK_angles[:, i], label='IK '+joints[i])
        #plt.ylim(-20, 180)
        plt.legend()
        if i == 0:
            plt.title("IK joint angles")
    if len(add_IK_joints) > 0:
        plt.figure()
        for i in range(len(add_IK_joints)):
            plt.subplot(len(add_IK_joints), 1, i + 1)
            plt.plot(IK_time, IK_angles[:, i+len(joints)], label='IK ' + add_IK_joints[i])
            #plt.ylim(-20, 180)
            plt.legend()
            if i == 0:
                plt.title("IK joint angles")
    plt.show()


def plot_cmc_reserve(cmc_force_file, cmc_error_file, n_res):
    """
    Evaluate osim CMC results by plotting reserve forces and joint errors
    INPUTS: - cmc_force_file: string, path of osim CMC force file
            - cmc_error_file: string, path of osim CMC error file
            - n_res:
    """
    # CMC reserve forces
    cmc_data = open(cmc_force_file, 'r')
    lines = cmc_data.readlines()
    res_forces = np.zeros((len(lines[23:]), n_res))
    cmc_time = np.zeros(len(lines[23:]))
    for l in range(len(lines[23:])):
        cmc_time[l] = lines[l+23].split()[0]
        for i in range(n_res):
            res_forces[l, i] = lines[l+23].split()[-i-1]

    plt.figure()
    for i in range(n_res):
        plt.plot(cmc_time, res_forces[:,i], label=lines[22].split()[-i-1])
    plt.legend()
    plt.title('CMC reserve forces')

    # CMC joint errors
    err_data = open(cmc_error_file, 'r')
    lines = err_data.readlines()
    err = np.zeros((len(lines[7:]), n_res))
    time = np.zeros(len(lines[7:]))
    for l in range(len(lines[7:])):
        time[l] = lines[l+7].split()[0]
        for i in range(n_res):
            err[l, i] = lines[l+7].split()[i+1]

    plt.figure()
    for i in range(n_res):
        plt.plot(time, err[:, i], label=lines[6].split()[i+1])
    plt.legend()
    plt.title('CMC joint errors')


def comp_cmc_emg(cmc_states_file, emg_folder, ti, tf, delay_video, disabled_muscles, n_emg=8, emg_step=0.5):
    """
    Compare osim CMC muscle activation (and muscle excitation if provided) with EMG data by plotting them
    INPUTS: - cmc_states_file: string, path of osim CMC states file
            - emg_folder: string, path to folder where EMG data files are
            - ti: float, movement initial time in video
            - tf: float, movement final time in video
            - delay_video: float, delay between video and emg data
            - disabled_muscles: string array, disabled muscle in osim model
            - n_emg: int, number of emg per plot
            - emg_step: float, y step between each emg on plots
    """
    emg_file = emg_folder + 'EMG_norm_data.txt'
    muscle_file = emg_folder + 'EMG_muscles.txt'
    time_file = emg_folder + 'EMG_time.txt'
    # CMC data
    cmc_data = open(cmc_states_file, 'r')
    lines = cmc_data.readlines()
    cmc_muscles = []
    for s in lines[6].split()[1:]:
        if s.split('/')[-1] == 'activation':
            if s.split('/')[-2] not in disabled_muscles:
                cmc_muscles.append(s.split('/')[-2])
    muscles_i = np.zeros(len(cmc_muscles))
    for s in range(len(lines[6].split()[1:])):
        if len(lines[6].split()[s + 1].split('/')) > 2:
            m = lines[6].split()[s + 1].split('/')[-2]
            if m in cmc_muscles:
                muscles_i[int(cmc_muscles.index(m))] = s + 1
    muscles_i = muscles_i.astype(int)

    cmc_acti = np.zeros((len(lines[7:]), len(cmc_muscles)))
    cmc_time = np.zeros(len(lines[7:]))
    for l in range(len(lines[7:])):
        cmc_time[l] = lines[l + 7].split()[0]
        for i in range(len(cmc_muscles)):
            cmc_acti[l, i] = lines[l + 7].split()[muscles_i[i]]

    # EMG data
    emg_periods = emg_data(emg_file, muscle_file, time_file, ti, tf, delay_video, plot=True)

    # Plot
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    for p in range(len(cmc_muscles) // n_emg + 1):
        plt.figure()
        for i in range(p * n_emg, min((p + 1) * n_emg, len(cmc_muscles))):
            plt.plot(cmc_time, cmc_acti[:, i] + (i - p * n_emg) * emg_step, color=colors[i - p * n_emg],
                     label=cmc_muscles[i])
            if cmc_muscles[i] in list(emg_periods.keys()):
                plt.plot(emg_periods[cmc_muscles[i]][0], emg_periods[cmc_muscles[i]][1] + (i - p * n_emg) * emg_step,
                         '--', color=colors[i - p * n_emg], label='ref ' + cmc_muscles[i])
            plt.legend()


if __name__ == '__main__':
        main()

