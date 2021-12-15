import numpy as np
import xlrd
import matplotlib.pyplot as plt
import os
from emg_control import emg_data
import pandas as pd
from opensim_environment import OsimModel

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
    model_name = 'UP001'
    kin_folder = 'C:/Users/Acer/Desktop/Pour alice/UP001/Tracked_cam/'
    emg_folder = 'C:/Users/Acer/Desktop/Pour alice/UP001/emg/'

    movement = 'scaling'
    IK_file = None  # None or path (kin_folder+movement+'/joints_IK_'+movement+'.mot')
    delay_video = 0

    if model_name == 'UP001':
        if movement == 'scaling':
            # Static kin data for scaling
            kin_file = kin_folder + 'df_cam1_916512060805_record_17_11_2021_1449_36.csv'  # 'df_cam1_816612061599_record_17_11_2021_1448_46.csv'
            results_folder = 'C:/Users/Acer/Desktop/Pour alice/UP001/kinematics/' + movement + '/'
            ti = 0  # in sec
            tf = 10
            markers = {'shoulder': 'left_shoulder', 'elbow': 'left_elbow', 'wrist': 'left_wrist'}
                       #'shoulder_r': 'right_shoulder', 'elbow_r': 'right_elbow', 'wrist_r': 'right_wrist'}
            smoothing = 3
            scaling_kin_data(kin_file, results_folder, ti, tf, markers, smoothing=smoothing, plot=True)

        else:
            # Movement kin data
            ti = 4.0
            tf = 4.48
            smoothing = 19
            joint_angles = scaled_kin_data(kin_folder, movement, ti, tf, markers, static_scaling, static_ground_ref,
                                           side='left', smoothing=smoothing, plot=True)

            IK_file = 'C:/Users/Acer/Desktop/Pour alice/UP001/kinematics/sh_flexion/very_smooth_IK_sh_flexion.mot'
            ti = 28.7
            tf = 34.1
            smoothing = 7
            txt_path = 'C:/Users/Acer/Desktop/Pour alice/UP001/kinematics/sh_flexion/29_34'
            ros_txt_file(IK_file, ti, tf, txt_path, smoothing=smoothing, joints=['shoulder_elev_l', 'elbow_flexion_l'])

            #if IK_file is not None:
            #    comp_IK_joints(IK_file, joint_angles, ['elv_angle', 'shoulder_rot'])

    plt.show()


def ros_txt_file(IK_file, ti, tf, txt_path, smoothing=None, joints=['shoulder_elev_l', 'elbow_flexion_l']):
    # IK data!
    IK_data = open(IK_file, 'r')
    lines = IK_data.readlines()
    IK_joints = lines[10].split()[1:]
    joints_i = np.zeros(len(joints))
    for i in range(len(joints)):
        joints_i[i] = IK_joints.index(joints[i]) + 1
    joints_i = joints_i.astype(int)
    IK_angles = np.zeros((len(lines[11:]), len(joints)))
    IK_time = np.zeros(len(lines[11:]))
    for l in range(len(lines[11:])):
        IK_time[l] = lines[l + 11].split()[0]
        for i in range(len(joints)):
            IK_angles[l, i] = lines[l + 11].split()[joints_i[i]]

    dt = IK_time[1]-IK_time[0]
    IK_vel = (IK_angles[1:,:] - IK_angles[:-1,:])/dt
    if smoothing is not None:
        IK_vel = moving_average(IK_vel, smoothing)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(IK_time, IK_angles[:,0], 'r', label='sh pos')
    axs[0].plot(IK_time, IK_angles[:, 1], 'b', label='elb pos')
    axs[1].plot(IK_time[:-1], IK_vel[:,0], '--r', label='sh vel')
    axs[1].plot(IK_time[:-1], IK_vel[:, 1], '--b', label='elb vel')
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel('angle [°]')
    axs[1].set_ylabel('vel [°/s]')
    axs[1].set_xlabel('time')
    axs[0].set_title('Joints kinematics from IK')
    plt.show()

    pos_traj = IK_angles[np.logical_and(IK_time >= ti, IK_time <= tf), :]
    vel_traj = IK_vel[np.logical_and(IK_time[:-1]>=ti, IK_time[:-1]<=tf), :]
    frames = int((tf-ti)*1500/3)
    ros_pos_traj = np.zeros((frames, 2))
    ros_vel_traj = np.zeros((frames, 2))
    time = np.linspace(ti, tf, len(pos_traj))
    vel_time = np.linspace(ti, tf, len(vel_traj))
    ros_time = np.linspace(ti, tf, frames)
    ros_pos_traj[:, 0] = np.interp(ros_time, time, pos_traj[:, 0])
    ros_pos_traj[:, 1] = np.interp(ros_time, time, pos_traj[:, 1])
    ros_vel_traj[:, 0] = np.interp(ros_time, vel_time, vel_traj[:, 0])
    ros_vel_traj[:, 1] = np.interp(ros_time, vel_time, vel_traj[:, 1])
    plt.figure()
    plt.plot(ros_pos_traj[:, 0], 'r', label='sh pos')
    plt.plot(ros_pos_traj[:, 1], 'b', label='elb pos')
    plt.plot(ros_vel_traj[:, 0], '--r', label='sh vel')
    plt.plot(ros_vel_traj[:, 1], '--b', label='elb vel')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('angle [°], vel [°/s]')
    plt.title('Trajectory profiles')
    plt.show()

    pos_file = open(txt_path+'_position.txt', 'w')
    vel_file = open(txt_path + '_velocity.txt', 'w')
    for l in range(len(ros_time)):
            pos_file.write(str(ros_pos_traj[l, 0]*3.14/180) + ' ' + str(ros_pos_traj[l, 1]*3.14/180) +'\n')
            vel_file.write(str(ros_vel_traj[l, 0]*3.14/180) + ' ' + str(ros_vel_traj[l, 1]*3.14/180) + '\n')


def scaling_kin_data(kin_file, kin_folder, ti, tf, markers, time_step=0.0333, smoothing=None, plot=False):
    """
    Extract kinematics data of 'static' period of interest from kin file .xls
    Write extracted data in osim format .trc in 'kin_folder/movement/' folder
    INPUTS: - kin_file: string, path of the kin file .xls
            - kin_folder: string, name of the folder to save results
            - ti: float, static period initial time in video
            - tf: float, static period final time in video
            - markers: dictionnary, markers name and corresponding pifpaf point to save in osim format .trc
            - time_step: float, openpifpaf time step
            - plot: bool, to plot or not the extracted data
    OUTPUTS: - write trc file
    """
    if not os.path.isdir(kin_folder):
        os.mkdir(kin_folder)

    # open file for reading
    df = pd.read_table(kin_file, sep=",")
    # data to extract
    if tf == -1 :
        tf = df['timestamp'][df.shape[0]-1]
    index_i = np.where(df['timestamp'] >= ti)[0][0]
    index_f = np.where(df['timestamp'] <= tf)[0][-1]
    time = np.zeros(index_f-index_i)
    markers_data = np.zeros((index_f-index_i, 3*len(markers)))
    for t in range(index_i, index_f):
            time[t-index_i] = df['timestamp'][t] - ti
            for m in range(len(markers)):
                markers_data[t-index_i, 3*m] = df[list(markers.values())[m]+'.x'][t]
                markers_data[t-index_i, 3 * m+1] = df[list(markers.values())[m] + '.y'][t]
                markers_data[t-index_i, 3 * m+2] = df[list(markers.values())[m] + '.z'][t]

    # write kin file
    dt = time[1] - time[0]
    kin_file = open(kin_folder+ 'kinematics.trc', 'w')
    kin_file.write('PathFileType\t4\t(X/Y/Z)\t kinematics .trc\n')
    kin_file.write(
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    kin_file.write(str(float(1/dt))+'\t'+str(float(1/dt))+'\t' + str(len(time)) + '\t'+str(len(markers))+'\tmm\t'+str(float(1/dt))+'\t1\t' + str(len(time)) + '\n')
    kin_file.write('Frame#\tTime\t')
    for m in list(markers.keys()):
        kin_file.write(m + '\t\t\t')
    kin_file.write('\n')
    kin_file.write('\t\t')
    for i in range(len(list(markers.keys()))):
        kin_file.write('X' + str(i + 1) + '\tY' + str(i + 1) + '\tZ' + str(i + 1) + '\t')
    kin_file.write('\n')
    kin_file.write('\n')
    frame = 1
    for t in range(len(time)):
        line = str(frame) + '\t' + str(time[t])
        for i in range(3 * len(markers)):
            line += '\t' + str(markers_data[t, i] * 1000)
        line += '\n'
        kin_file.write(line)
        frame += 1

    if plot:
        plt.figure()
        for m in range(len(list(markers.keys()))):
            plt.plot(time, markers_data[:, 3 * m], label=list(markers.keys())[m] + '_x')
            plt.plot(time, markers_data[:, 3 * m + 1], label=list(markers.keys())[m] + '_y')
            plt.plot(time, markers_data[:, 3 * m + 2], label=list(markers.keys())[m] + '_z')
        plt.legend()
        plt.show()

    if smoothing is not None:
        smooth_kin_data = moving_average(markers_data, smoothing)
        smooth_time = time
        # write kin file
        kin_file = open(kin_folder + 'smooth_kinematics.trc', 'w')
        kin_file.write('PathFileType\t4\t(X/Y/Z)\t smooth kinematics .trc\n')
        kin_file.write(
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        kin_file.write(str(float(1/dt))+'\t'+str(float(1/dt))+'\t'+str(len(smooth_time)) + '\t'+str(len(markers))+
                       '\tmm\t'+str(float(1/dt))+'\t1\t' + str(len(smooth_time)) + '\n')
        kin_file.write('Frame#\tTime\t')
        for m in markers:
            kin_file.write(m + '\t\t\t')
        kin_file.write('\n')
        kin_file.write('\t\t')
        for i in range(len(list(markers.keys()))):
            kin_file.write('X' + str(i + 1) + '\tY' + str(i + 1) + '\tZ' + str(i + 1) + '\t')
        kin_file.write('\n')
        kin_file.write('\n')
        frame = 1
        for t in range(len(smooth_time)):
            line = str(frame) + '\t' + str(smooth_time[t])
            for i in range(3 * len(markers)):
                line += '\t' + str(smooth_kin_data[t, i] * 1000)
            line += '\n'
            kin_file.write(line)
            frame += 1

        # plot smooth kin data
        if plot:
            plt.figure()
            for m in range(len(list(markers.keys()))):
                plt.plot(smooth_time, smooth_kin_data[:, 3*m], label=list(markers.keys())[m]+'_x')
                plt.plot(smooth_time, smooth_kin_data[:, 3*m+1], label=list(markers.keys())[m]+'_y')
                plt.plot(smooth_time, smooth_kin_data[:, 3 * m+2], label=list(markers.keys())[m] + '_z')
            plt.legend()
            plt.show()


def moving_average(x, w):
    smooth_x = x
    if len(np.shape(x)) > 1:
        n_m = np.shape(x)[1]
        for m in range(n_m):
            val = np.concatenate((x[0, m]*np.ones((w-1)//2), x[:, m], x[-1, m]*np.ones((w-1)//2)))
            smooth_x[:, m] = np.convolve(val, np.ones(w), 'valid') / w
    else:
        val = np.concatenate((x[0] * np.ones((w - 1) // 2), x[:], x[-1] * np.ones((w - 1) // 2)))
        smooth_x[:] = np.convolve(val, np.ones(w), 'valid') / w
    return smooth_x


def scaled_kin_data(kin_folder, movement, ti, tf, markers, scaling, ground_ref, time_step=0.0333,
                    joints=['shoulder_elev', 'elbow_flexion'], side='right', smoothing=None, plot=False):
    """
    Extract kinematics data of 'movement 'of interest from 'kinematics.xls'
    Scale the data from previous 'static' scaling factor and compute joint angles of interest (print openpifpaf
        confidences)
    Save extracted data in osim format .trc and joint angles in osim format '.mot' in 'kin_folder/movement/' folder
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
        markers_i[i] = bodies.index(side+'_'+markers[i]+'_x')
    markers_i = markers_i.astype(int)

    period = [ti // 1 * 60 + ti % 1 * 100, tf // 1 * 60 + tf % 1 * 100]
    time = np.zeros(int((period[1]-period[0])/time_step))
    kin_data = np.zeros((int((period[1]-period[0])/time_step), len(markers)*3))
    confidence_data = np.zeros((int((period[1]-period[0])/time_step), len(markers)))

    for t in range(1, sheet.nrows):
        if float(sheet.cell_value(1, 0)) >= period[0]:
            t0 = 1
        elif float(sheet.cell_value(t, 0)) < period[0] and float(sheet.cell_value(t + 1, 0)) >= period[0]:
            t0 = t + 1
        if float(sheet.cell_value(t, 0)) >= period[0] and float(sheet.cell_value(t, 0)) <= period[1]:
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
    file_path.write('Joints_'+movement+'\nversion=1\nnRows='+str(len(joint_angles['time'])-1) +
                    '\nnColumns='+str(len(joints))+'\ninDegrees=yes\nendheader\n')
    file_path.write('time')
    for j in joints:
        file_path.write('\t'+j)
    file_path.write('\n')
    for t in range(len(joint_angles['time'])-1):
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
            plt.plot(kin_data[:-2,0], label='sx')
            plt.plot(kin_data[:-2,1], label='sy')
            plt.plot(kin_data[:-2,3], label='ex')
            plt.plot(kin_data[:-2,4], label='ey')
            plt.plot(kin_data[:-2,6], label='wx')
            plt.plot(kin_data[:-2,7], label='wy')
            plt.legend()
            plt.show()
            # joint angles
            plt.figure()
            plt.plot(joint_angles['time'][:-2], joint_angles['shoulder_elev'][:-2], label="sh_elev")
            plt.plot(joint_angles['time'][:-2], joint_angles['elbow_flexion'][:-2], label='elb_flexion')
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
    kin_file.write('30.00\t30.00\t'+str(len(time)-2)+'\t3\tmm\t30.00\t1\t'+str(len(time)-2)+'\n')
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
    for t in range(len(time)-2):
        line = str(frame)+'\t'+str(time[t])
        for i in range(3*len(markers)):
            line += '\t'+str(kin_data[t, i]*1000)
        line += '\n'
        kin_file.write(line)
        frame += 1

    if smoothing is not None:
        smooth_kin_data = moving_average(kin_data[:-2, :], smoothing)
        smooth_time = time[:-2]
        # write kin file
        kin_file = open(kin_folder+movement+"/smooth_kin_"+movement+".trc", 'w')
        kin_file.write('PathFileType\t4\t(X/Y/Z)\t' + movement + '.trc\n')
        kin_file.write(
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        kin_file.write('30.00\t30.00\t' + str(len(smooth_time)) + '\t3\tmm\t30.00\t1\t' + str(len(smooth_time)) + '\n')
        kin_file.write('Frame#\tTime\t')
        for m in markers:
            kin_file.write(m + '\t\t\t')
        kin_file.write('\n')
        kin_file.write('\t\t')
        for i in range(len(markers)):
            kin_file.write('X' + str(i + 1) + '\tY' + str(i + 1) + '\tZ' + str(i + 1) + '\t')
        kin_file.write('\n')
        kin_file.write('\n')
        frame = 1
        for t in range(len(smooth_time)):
            line = str(frame) + '\t' + str(smooth_time[t])
            for i in range(3 * len(markers)):
                line += '\t' + str(smooth_kin_data[t, i] * 1000)
            line += '\n'
            kin_file.write(line)
            frame += 1

        # plot smooth kin data
        if plot:
            plt.figure()
            plt.plot(smooth_kin_data[:, 0], 'b', label='sx')
            plt.plot(smooth_kin_data[:, 1], '--b', label='sy')
            plt.plot(smooth_kin_data[:, 3], 'c', label='ex')
            plt.plot(smooth_kin_data[:, 4], '--c', label='ey')
            plt.plot(smooth_kin_data[:, 6], color='orange', label='wx')
            plt.plot(smooth_kin_data[:, 7], linestyle='dashed', color='orange', label='wy')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('position [m]')
            plt.title('Markers position from tracking')
            plt.show()

            wrist_pos = np.sqrt(smooth_kin_data[:-2, 6] * smooth_kin_data[:-2, 6] + smooth_kin_data[:-2, 7] * smooth_kin_data[:-2, 7])
            #wrist_pos = moving_average(wrist_pos, smoothing)
            dt = time[1] - time[0]
            wrist_vel = (wrist_pos[1:] - wrist_pos[:-1]) / dt
            wrist_vel = moving_average(wrist_vel, 5)
            plt.figure()
            plt.plot(wrist_pos, 'b', label='w pos')
            plt.plot(wrist_vel, '--b', label='w vel')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('position [m], vel [m/s]')
            plt.title('Wrist kinematics')
            plt.show()

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


def alex_kin(alex_file, ti=0, unit='sec',
             ref={'R_ang_pos_5': 'shoulder_elev', 'R_ang_pos_1': 'shoulder_add', 'R_ang_pos_2': 'shoulder_rot',
                  'R_ang_pos_6': 'elbow_flexion', 'R_pressure': 'pression'}, plot=False):

    if unit == 'min':
        ti = ti // 1 * 60 + ti % 1 * 100

    with open(alex_file, 'r') as f1:
        lines = f1.readlines()
        values = np.zeros((len(lines) - 2, len(ref)))
        time = np.zeros(len(lines) - 2)
        joints = lines[0].split(',')
        for r in range(2, len(lines)):
            for j in range(len(joints)):
                if joints[j] in list(ref.keys()):
                    time[r - 2] = float(lines[r].split(',')[3])
                    if list(ref.values())[list(ref.keys()).index(joints[j])] == 'elbow_flexion':
                        values[r - 2, list(ref.keys()).index(joints[j])] = 90 + float(
                            lines[r].split(',')[j]) * 180 / 3.14
                    else:
                        values[r - 2, list(ref.keys()).index(joints[j])] = float(
                            lines[r].split(',')[j]) * 180 / 3.14

    if plot:
        plt.figure()
        for j in range(len(ref)):
            plt.plot(time[time > ti] - time[time > ti][0], values[time > ti, j], label=list(ref.values())[j])
        plt.legend()
        plt.xlabel('time [s]')

    return values[time > ti, :], time[time > ti] - time[time > ti][0], ref


if __name__ == '__main__':
        main()

