#!/usr/bin/env python

from opensim_environment import OsimModel
import numpy as np
import matplotlib.pyplot as plt
import os
from osim_model import *


def main():
    """
    Script to evaluate EMG to kinematics data:
    Simulate osim model with EMG data of a 'movement' of interest as muscles activation
    Compare resulting joint angles with osim IK ones by plotting them, simulation states and joint angles plots are
    saved in 'save_folder'
    """

    #: Osim model
    osim_file = 'models/Henri.osim'
    osim_file = modify_Millard(osim_file, ignore_tendon='true', fiber_damping=0.001, ignore_dyn='true')  # set
                               # ignore_dyn to 'true' to provide EMG data as muscle activation
    #osim_file = lock_Coord(osim_file, ['shoulder_rot', 'elv_angle'], 'false')
    osim_file = modify_default_Coord(osim_file, 'elv_angle', 1.2)
    osim_file = modify_default_Coord(osim_file, 'r_shoulder_elev', 0.14)
    osim_file = modify_default_Coord(osim_file, 'shoulder_rot', 0.16)
    osim_file = modify_default_Coord(osim_file, 'elbow_flexion', 0.29)

    #: EMG control
    emg_folder = 'emg/'
    movement = 'sh_adduction'
    ti = 2.50
    tf = 3.02
    delay_video = 108.4 - 55.48

    # Plot
    IK_ref_file = None  # None or 'kinematics/' + movement + "/joints_IK_" + movement + ".mot"
    coord_plot = ["shoulder_elev", "elbow_flexion", "elv_angle", 'shoulder_rot']
    save_folder = emg_folder + movement + '/'

    step_size = 0.01
    integ_acc = 0.0001
    period = np.array([ti // 1 * 60 + ti % 1 * 100, tf // 1 * 60 + tf % 1 * 100])
    n_steps = int((period[1]-period[0])/step_size)
    emg_period = emg_data(emg_folder, ti, tf, delay_video, plot=True)
    if movement in ['sh_flexion', 'elb_flexion', 'arm_flexion']:
        osim_file = modify_default_Coord(osim_file, 'elv_angle', 1.574)
    elif movement in ['sh_adduction']:
        osim_file = modify_default_Coord(osim_file, 'elv_angle', 0.0)

    osim_control(osim_file, step_size, n_steps, emg_period, integ_acc=integ_acc, coord_plot=coord_plot,
                 IK_ref_file=IK_ref_file, save_folder=save_folder, visualize=True, show_plot=True, save_kin=True)


def osim_control(osim_file, step_size, n_steps, emg_period, integ_acc=0.0001,
                 coord_plot=['shoulder_elev', 'elbow_flexion'], IK_ref_file=None, save_folder="results/",
                 visualize=False, show_plot=False, save_kin=False):
    """
    Simulate osim model with EMG data of a movement of interest as muscle activation
    Compare the resulting joint angles with osim IK ones by plotting them
    Save simulation states and joint angles plots in save_folder
    INPUTS: - osim_file: string, path to osim model
            - step_size: float, osim integration step size
            - n_steps: int, osim number of integration steps
            - emg_period: dictionary, EMG data of each muscle ({'TRIlong': float array, ...})
            - integ_acc: osim integration accuracy
            - coord_plot: None or string array, coordinates to plot
            - IK_ref_file: None or string, path to osim IK joint angles file
            - save_folder: string, path to folder where simulation states and plots will be saved
            - visualize: bool, to visualize or not osim simulation
            - show_plot: bool, to show or not the plots
            - save_kin: bool, to save or not tehe simulation states
    """

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    #: Osim model
    model = OsimModel(osim_file, step_size, integ_acc, body_ext_force=None, visualize=visualize, save_kin=save_kin)
    n_muscles = model.model.getMuscles().getSize()
    muscle_names = [None]*n_muscles
    for i in range(n_muscles):
        muscle_names[i] = model.model.getMuscles().get(i).getName()

    #: Controls
    controls = def_controls(emg_period, muscle_names, n_steps, step_size, plot=False)

    #: Initialize
    model.reset()
    model.reset_manager()

    #: States to plot
    if coord_plot is not None:
        coord_states = np.zeros((len(coord_plot), 2, n_steps-1))

    #: Integrate step 0
    #: Actuate the model from control outputs
    model.actuate(np.zeros(n_muscles))
    #: Integration musculoskeletal system
    model.integrate()

    #: Integrate
    for j in range(1, n_steps):

        #: Actuate the model from control outputs
        model.actuate(controls[:, j])

        #: Integration musculoskeletal system
        model.integrate()

        #: Results from the integration
        res = model.get_state_dict()

        if coord_plot is not None:
            for i in range(len(coord_plot)):
                coord_states[i, 0, j-1] = res["coordinate_pos"][coord_plot[i]]
                coord_states[i, 1, j-1] = res["coordinate_vel"][coord_plot[i]]

    if IK_ref_file is not None:
        IK_ref = open(IK_ref_file, 'r')
        lines = IK_ref.readlines()
        IK_joints = lines[10].split()[1:]
        IK_ref = np.zeros((len(lines[11:]), len(IK_joints)+1))
        for l in range(len(lines[11:])):
            IK_ref[l, 0] = lines[l+11].split()[0]
            for j in range(len(IK_joints)):
                IK_ref[l, j+1] = lines[l+11].split()[j+1]

    #: Coordinates plot
    time = np.arange(n_steps-1)*step_size
    if coord_plot is not None:
        plt.figure("coord plot")
        for i in range(max(len(coord_plot), 2)):
            ax = plt.subplot(len(coord_plot[:2]), 1, i+1)
            ax.plot(time, coord_states[i, 0]*180/3.14, 'b', label=coord_plot[i]+" angle", )
            if IK_ref_file is not None and coord_plot[i] in IK_joints:
                ax.plot(IK_ref[:, 0], IK_ref[:, IK_joints.index(coord_plot[i])+1], '--b',
                        label=coord_plot[i] + " ref", )
            ax.legend()
            ax.set_ylabel("angle [°]", color='b')
            ax.set_xlabel("time [s]")
            #ax.set_ylim(-20, 160)
            plt.title(coord_plot[i]+" kinematics")
        plt.savefig(save_folder+"coord_plot")

    if len(coord_plot) > 2:
        plt.figure("coord plot 2")
        for i in range(2, len(coord_plot)):
            ax = plt.subplot(len(coord_plot[2:]), 1, i - 1)
            ax.plot(time, coord_states[i, 0] * 180 / 3.14, 'b', label=coord_plot[i] + " angle", )
            if IK_ref_file is not None and coord_plot[i] in IK_joints:
                ax.plot(IK_ref[:, 0], IK_ref[:, IK_joints.index(coord_plot[i]) + 1], '--b',
                        label=coord_plot[i] + " IK ref", )
            ax.legend()
            ax.set_ylabel("angle [°]", color='b')
            ax.set_xlabel("time [s]")
            plt.title(coord_plot[i] + " kinematics")
        plt.savefig(save_folder + "coord_IK_plot")

    if show_plot:
        plt.show()
    plt.close('all')

    #: Save simulation states
    model.save_simulation(save_folder)


def def_controls(emg_period, muscle_names, n_steps, step_size, plot=False):
    """
    Build simulation controls array for all osim model muscles from EMG dictionary
    INPUTS: - emg_period: dictionary, EMG data of each muscle ({'TRIlong': float array, ...)}
            - muscle_names: string array, osim model muscle names
            - n_steps: int, osim number of integration steps
            - step_size: float, osim integration step size
            - plot: bool, to plot or not the controls
    OUTPUT: - controls: float array, simulation controls for all osim model muscles
    """

    n_muscles = len(muscle_names)
    controls = np.zeros((n_muscles, n_steps))
    time = np.linspace(0, step_size*n_steps, n_steps)
    emg_muscle_names = list(emg_period.keys())
    for m in emg_muscle_names:
        if m in muscle_names:
            controls[muscle_names.index(m)] = np.interp(time, emg_period[m][0], emg_period[m][1])

    # Plot
    if plot:
        plt.figure()
        for m in range(len(emg_muscle_names)):
            if emg_muscle_names[m] in muscle_names:
                plt.plot(controls[muscle_names.index(emg_muscle_names[m])]+m*0.1, label=emg_muscle_names[m])
        plt.legend()
        plt.show()

    return controls


def emg_data(emg_folder, ti, tf, delay_video, plot=False):
    """
    Extract EMG data of a movement of interest
    INPUTS: - emg_folder: string, path to folder where EMG files are
            - ti: float, movement initial time in video
            - tf: float, movement final time in video
            - delay_video: float, delay between the video and EMG data
            - plot: bool, to plot or not the EMG data
    OUTPUT: - emg_period: dictionary, EMG data of each muscle ({'TRIlong': float array, ...})
    """
    # EMG data
    emg_file = emg_folder+'EMG_norm_data.txt'
    muscle_file = emg_folder+'EMG_muscles.txt'
    time_file = emg_folder+'EMG_time.txt'
    osim_muscle_names = ['THE', 'EPB', 'FDSI', 'FPL', 'ECRL', 'PT', 'FCR', 'BRA', 'FDSM', 'BRD', 'EDCM', 'FCU', 'ECU',
                         'BIClong', 'TRIlat', 'TRIlong', 'DELT1', 'DELT2', 'DELT3', 'PECM1', 'INFSP', 'SUPSP', 'RH',
                         'TRAPC', 'BICshort']
    period = np.array([ti // 1 * 60 + ti % 1 * 100, tf // 1 * 60 + tf % 1 * 100]) - delay_video
    emg_lines = open(emg_file, 'r').readlines()
    muscle_names = open(muscle_file, 'r').readline().split()
    time_lines = open(time_file, 'r').readlines()
    emg_data = np.zeros((len(emg_lines), len(muscle_names)))
    emg_time = np.zeros((len(emg_lines), len(muscle_names)))
    for l in range(len(emg_lines)):
        for m in range(len(muscle_names)):
            emg_data[l, m] = float(emg_lines[l].split()[m])
            emg_time[l, m] = float(time_lines[l].split()[m])

    emg_period = {}
    for m in range(len(muscle_names)):
        time_i = emg_time[emg_time[:, m] > period[0], m]
        time = time_i[time_i < period[1]]
        time = time - time[0]
        data = emg_data[emg_time[:, m] > period[0], m]
        data = data[time_i < period[1]]
        emg_period[osim_muscle_names[m]] = [time, data]

    # Plot
    if plot:
        plt.figure()
        for m in range(len(osim_muscle_names)):
            plt.plot(emg_period.get(osim_muscle_names[m])[0], emg_period.get(osim_muscle_names[m])[1] + m*0.1,
                     label=osim_muscle_names[m])
        plt.legend()
        plt.show()

    return emg_period


if __name__ == '__main__':
        main()