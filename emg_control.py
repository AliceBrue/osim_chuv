#!/usr/bin/env python

from opensim_environment import OsimModel
from opensim import *
import numpy as np
import matplotlib.pyplot as plt
import os
from osim_model import *
from kinematics import *
import pandas as pd


def main():
    """
    Script to evaluate EMG to kinematics data:
    Simulate osim model with EMG data of a 'movement' of interest as muscles activation
    Compare resulting joint angles with osim IK ones by plotting them, simulation states and joint angles plots are
    saved in 'save_folder'
    """

    case = 'NM00'  # 'Henri' or 'NM00'

    if case == 'Henri':
        #: Osim model
        osim_file = 'models/Henri.osim'
        osim_file = modify_Millard(osim_file, ignore_tendon='true', fiber_damping=0.001, ignore_dyn='true')
                                   # ignore_dyn to 'true' to provide EMG data as muscle activation
        osim_file = lock_Coord(osim_file, ['shoulder_rot', 'elv_angle'], 'false')
        # Manual init joint angles  ## TO DO init from pifpaf file"
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
        emg_period = emg_data(emg_folder, ti, tf, unit='min', delay_video=delay_video, plot=True)
        if movement in ['sh_flexion', 'elb_flexion', 'arm_flexion']:
            osim_file = modify_default_Coord(osim_file, 'elv_angle', 1.574)
        elif movement in ['sh_adduction']:
            osim_file = modify_default_Coord(osim_file, 'elv_angle', 0.0)

        osim_control(osim_file, step_size, n_steps, emg_period, integ_acc=integ_acc, coord_plot=coord_plot,
                     IK_ref_file=IK_ref_file, save_folder=save_folder, visualize=True, show_plot=True, save_kin=True)

    elif case == 'NM00':

        # Patient
        id = 4
        session = 1
        recording = 'Run_number_95_Plot_and_Store_Rep_1.0_1056.mat'
        n_stim = 12
        alex_file = 'C:/Users/Acer/Desktop/Pour alice/NM00' + str(id) + '/ALEx20210303114202.csv'  # path or None
        alex_side = 'L'

        osim_file = 'models/alex_marker.osim'  # '_elbk.osim
        osim_file = modify_elbow_k(osim_file, k=1, min_angle=90, transition=50)   # default k=1e-08, min_angle=190, transtion=1
        osim_file = modify_Millard(osim_file, 'true', 0.001, 'true')  # ignore tendon comp, fiber damping, activation dynamics
        # Model spastic muscle
        spas = False
        if spas:
            spas_muscles = ['BIClong']
            min_acti = [0.5]  # default 0.01
            osim_file = min_acti_Muscle(osim_file, spas_muscles, min_acti)
        osim_file = lock_Coord(osim_file, ['shoulder_elev', 'elv_angle', 'shoulder_rot', 'elbow_flexion'], 'true')
        step_size = 0.01
        integ_acc = 0.001

        #: EMG control
        emg_file = 'C:/Users/Acer/Desktop/Pour alice/NM00' + str(id) + '/Data/norm_max_emg/' + recording +\
                   '/stim_'+str(n_stim)+'/EMG_norm_data.txt'
        muscle_file = 'C:/Users/Acer/Desktop/Pour alice/NM00' + str(id) + '/emg_name.txt'
        time_file = 'C:/Users/Acer/Desktop/Pour alice/NM00' + str(id) + '/Data/norm_max_emg/' + recording +\
                    '/stim_'+str(n_stim)+'/EMG_time.txt'
        osim_muscle_names_004 = ['FCR', 'FCU', 'ED', 'FDSM', 'BRA', 'BIClong', 'TRIlong', 'DELT1', 'DELT2', 'DELT3',
                                'BRD', 'PECM1', 'INFSP', 'SUPSP', 'RH']
        osim_muscle_names_005_1 = ['THE', 'BRD', 'ECRL', 'FDSM', 'SUPSP', 'DELT2', 'DELT1', 'DELT3', 'INFSP', 'TRIlong',
                                 'EDCM', 'TMAJ', 'PECM1', 'BICshort', 'FCU']
        osim_muscle_names_005_2 = ['ECRL', 'BRD', 'FDSM', 'ED', 'DELT2', 'DELT1', 'DELT3', 'BIClong', 'TRIlong',
                                 'PECM1', 'LevScap', 'FCU', 'sync']
        if id == 4:
            osim_muscle_names = osim_muscle_names_004
        elif id == 5 and session == 1:
            osim_muscle_names = osim_muscle_names_005_1
        elif id == 5 and session == 2:
            osim_muscle_names = osim_muscle_names_005_2

        ti = 0  # in sec
        tf = -1  # -1 to simulate entire stim period
        emg_period, period = emg_data(emg_file, muscle_file, time_file, ti, tf, unit='sec', factor=1, plot=True,
                                      osim_muscle_names=osim_muscle_names)
        n_steps = int((period[1] - period[0]) / step_size)

        # Init joint angles from Alex
        if alex_file is not None:
            alex_ref = {alex_side+'_ang_pos_5': 'shoulder_elev', alex_side+'_ang_pos_1': 'shoulder_add',
                        alex_side+'_ang_pos_2': 'shoulder_rot', alex_side+'_ang_pos_6': 'elbow_flexion',
                        alex_side+'_pressure': 'pression'}
            alex_j_values, alex_time, alex_ref = alex_kin(alex_file, ti=ti, unit='sec', ref=alex_ref, plot=False)
            osim_file = init_from_alex(osim_file, alex_file, ti, alex_side)
        else:
            alex_j_values, alex_time, alex_ref = None, None, None
        # Manual init
        osim_file = modify_default_Coord(osim_file, 'shoulder_elev', 0.15)
        osim_file = modify_default_Coord(osim_file, 'elv_angle', 1.3)
        osim_file = modify_default_Coord(osim_file, 'shoulder_rot', 0)
        osim_file = modify_default_Coord(osim_file, 'elbow_flexion', 1.5)

        #: Plots
        coord_plot = ["shoulder_elev", "elbow_flexion", "elv_angle", 'shoulder_rot']
        save_folder = 'C:/Users/Acer/Desktop/Pour alice/NM00' + str(id) + "/Results/" + "test/" #\
                      #str(recording.split('_')[2])+'_'+str(n_stim)+'/'

        alex_torques = None  # ['shoulder_elev', 'elbow_flexion', 'shoulder_add', 'shoulder_rot']
        alex_values = None  # compensation level in [0, 1]
        ref = {alex_side+'_Torque_3': 'shoulder_elev', alex_side+'_Torque_1': 'shoulder_add',
               alex_side+'_Torque_2': 'shoulder_rot', alex_side+'_Torque_4': 'elbow_flexion'}
        alex_t_values, alex_time, _ = alex_torque(alex_file, ti=ti, ref=ref, plot=False)  # to compare torques or None

        if alex_values is not None:
            save_folder = save_folder + 'T_' + str(alex_values) + '/'
        osim_alex_control(osim_file, step_size, n_steps, emg_period, emg_factor=1, integ_acc=integ_acc,
                          coord_plot=coord_plot, alex_torques=alex_torques, alex_values=alex_values,
                          alex_j_values=alex_j_values, alex_time=alex_time, alex_ref=alex_ref, alex_t_values=alex_t_values,
                          save_folder=save_folder, visualize=True, show_plot=True, save_kin=True)


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


def osim_alex_control(osim_file, step_size, n_steps, emg_period, emg_factor=1, integ_acc=0.0001, coord_plot=None, alex_j_values=None,
                      alex_torques=None, alex_values=None, alex_time=None, alex_ref=None, alex_t_values=None,
                      save_folder="results/", visualize=False, show_plot=False, save_kin=False):
    """ Integrate opensim model with muscle controls """

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    moments = None
    if alex_t_values is not None:
        moments = coord_plot
    #: Osim model
    model = OsimModel(osim_file, step_size, integ_acc, body_ext_force=None, alex_torques=alex_torques, moments=moments,
                      visualize=visualize, save_kin=save_kin)
    n_muscles = model.model.getMuscles().getSize()
    muscle_names = [None]*n_muscles
    for i in range(n_muscles):
        muscle_names[i] = model.model.getMuscles().get(i).getName()

    tf = n_steps * step_size

    #: Controls
    controls = def_controls(emg_period, muscle_names, n_steps, step_size, plot=False)

    #: Initialize
    model.reset()
    model.reset_manager()

    #: If Alex torques
    if alex_torques is not None:
        if not isinstance(alex_values, float):
            alex_controls = alex(alex_torques, alex_values, alex_time, alex_ref, step_size, n_steps)
        else:
            torques = np.zeros((n_steps - 1, 6))

    #: States to plot
    if coord_plot is not None:
        coord_states = np.zeros((len(coord_plot), 2, n_steps-1))
    if alex_t_values is not None:
        torque_states = np.zeros((len(coord_plot), n_steps - 1))

    # Markers
    sh = np.zeros((3, n_steps - 1))
    m1 = np.zeros((3, n_steps - 1))
    elb = np.zeros((3, n_steps-1))
    m2 = np.zeros((3, n_steps-1))

    #: Integrate step 0
    #: Actuate the model from control outputs
    model.actuate(np.zeros(n_muscles))
    #: Integration musculoskeletal system
    model.integrate()

    #: Integrate
    for j in range(1, n_steps):

        #: Actuate the model from control outputs
        model.actuate(controls[:, j]*emg_factor)

        if alex_torques is not None and j > 1:
            if not isinstance(alex_values, float):
                for p in range(len(alex_torques)):
                    t = alex_torques[p]
                    if t in ['shoulder_elev', 'elbow_flexion']:
                        dir = 2  # Z
                    elif t == 'shoulder_add':
                        dir = 0  # X
                    elif t == 'shoulder_rot':
                        dir = 1  # Y
                    ext_torque = opensim.PrescribedForce.safeDownCast(model.model.getForceSet().get('Alex_'+t))
                    ext_torque.set_forceIsGlobal(False)
                    torqueFunctionSet = ext_torque.get_torqueFunctions()
                    func = opensim.Constant.safeDownCast(torqueFunctionSet.get(dir))
                    func.setValue(alex_controls[t][j])
            else:
                ext_torque = opensim.PrescribedForce.safeDownCast(model.model.getForceSet().get('Alex_shoulder_elev'))
                ext_torque.set_forceIsGlobal(True)
                torqueFunctionSet = ext_torque.get_torqueFunctions()
                elb_torque = opensim.PrescribedForce.safeDownCast(model.model.getForceSet().get('Alex_elbow_flexion'))
                elb_torque.set_forceIsGlobal(True)
                elbtorqueFunctionSet = elb_torque.get_torqueFunctions()
                elb_t, sh_t = compute_alex_torque(sh[:, j-2], m1[:, j-2], elb[:, j-2], m2[:, j-2])
                for dir in range(3):
                    func = opensim.Constant.safeDownCast(torqueFunctionSet.get(dir))
                    func.setValue(-alex_values*sh_t[dir])
                    torques[j-1, dir] = -alex_values*sh_t[dir]
                    elbfunc = opensim.Constant.safeDownCast(elbtorqueFunctionSet.get(dir))
                    elbfunc.setValue(-alex_values*elb_t[dir])
                    torques[j - 1, 3+dir] = -alex_values*elb_t[dir]

        #: Integration musculoskeletal system
        model.integrate()

        #: Results from the integration
        res = model.get_state_dict()

        sh[:, j - 1] = res["markers"]['shoulder']['pos']
        m1[:, j - 1] = res["markers"]['m1']['pos']
        elb[:, j - 1] = res["markers"]['elbow']['pos']
        m2[:, j - 1] = res["markers"]['m2']['pos']

        if coord_plot is not None:
            for i in range(len(coord_plot)):
                coord_states[i, 0, j-1] = res["coordinate_pos"][coord_plot[i]]
                coord_states[i, 1, j-1] = res["coordinate_vel"][coord_plot[i]]
        if alex_t_values is not None:
            for i in range(len(coord_plot)):
                torque_states[i, j-1] = res['coordinate_muscle_moment_arm'][coord_plot[i]]

    #: Coordinates plot
    time = np.arange(n_steps-1)*step_size
    if coord_plot is not None:
        plt.figure("coord plot")
        n = min(2, len(coord_plot))
        for i in range(n):
            ax = plt.subplot(n, 1, i+1)
            ax.plot(time, coord_states[i, 0]*180/3.14, 'b', label=coord_plot[i]+" angle")
            if coord_plot[i] == 'shoulder_elev':
                if alex_j_values is not None and 'shoulder_add' in list(alex_ref.values()):
                    ax.plot(alex_time[alex_time<tf], alex_j_values[alex_time<tf,
                                                                   list(alex_ref.values()).index('shoulder_add')],
                            '--', color='grey', label="shoulder add ref")
            if alex_j_values is not None and coord_plot[i] in list(alex_ref.values()):
                ax.plot(alex_time[alex_time < tf], alex_j_values[alex_time < tf,
                                                                 list(alex_ref.values()).index(coord_plot[i])],
                        '--b', label=coord_plot[i] + " ref")
                ax.legend()
            ax.set_ylabel("angle [°]", color='b')
            ax.set_xlabel("time [s]")
            #ax.set_ylim(-20, 160)
            plt.title(coord_plot[i])
        plt.tight_layout()
        plt.savefig(save_folder+"coord_plot")

        if len(coord_states) > 2:
            plt.figure("coord plot 2")
            n = len(coord_states) - 2
            for i in range(n):
                ax = plt.subplot(n, 1, i + 1)
                ax.plot(time, coord_states[2+i, 0] * 180 / 3.14, 'b', label=coord_plot[2+i] + " angle")
                if alex_j_values is not None and coord_plot[2+i] in list(alex_ref.values()):
                    ax.plot(alex_time[alex_time < tf], alex_j_values[alex_time < tf,
                                                                     list(alex_ref.values()).index(coord_plot[2+i])],
                            '--b', label=coord_plot[2+i] + " ref")
                    ax.legend()
                ax.set_ylabel("angle [°]", color='b')
                ax.set_xlabel("time [s]")
                # ax.set_ylim(-20, 160)
                plt.title(coord_plot[2+i])
            plt.tight_layout()
            plt.savefig(save_folder + "coord_plot_2")

    if alex_torques is not None:
        if not isinstance(alex_values, float):
            if len(alex_torques) > 1 and 'elbow_flexion' in alex_torques:
                fig, ax = plt.subplots(2, 1)
                for i in range(len(alex_torques)):
                    if alex_torques[i] == 'elbow_flexion':
                        ax[1].plot(alex_time[alex_time<tf], alex_values[alex_time<tf, list(alex_ref.values()).index(alex_torques[i])],
                                'b', label=alex_torques[i])
                    else:
                        ax[0].plot(alex_time[alex_time<tf], alex_values[alex_time<tf, list(alex_ref.values()).index(alex_torques[i])]
                                   , label=alex_torques[i])
                    ax[1].set_ylabel("torque [Nm]", color='b')
                    ax[1].set_xlabel("time [s]")
                    #ax.set_ylim(-20, 160)
                    ax[1].set_title("Elbow torque")
                    ax[0].legend()
                    ax[0].set_ylabel("torque [Nm]", color='b')
                    ax[0].set_xlabel("time [s]")
                    # ax.set_ylim(-20, 160)
                    ax[0].set_title("Shoulder torques")
                plt.tight_layout()
                plt.savefig(save_folder+"alex_torques")
        else:
            fig, ax = plt.subplots(2, 1)
            labels = ['x', 'y', 'z']
            for dir in range(3):
                ax[0].plot(time, torques[:, dir], label=labels[dir])
                ax[1].plot(time, torques[:, 3+dir], label=labels[dir])
            ax[0].legend()
            ax[1].legend()
            ax[0].set_title('Shoulder torques')
            ax[1].set_title('Elbow torques')
            plt.savefig(save_folder + "torques")

    #: Save simulation states
    model.save_simulation(save_folder)

    # Torque comparison
    if alex_t_values is not None:
        plt.figure('comp torques')
        n = min(2, len(coord_plot))
        for i in range(n):
            ax = plt.subplot(n, 1, i + 1)
            ax.plot(time, torque_states[i], 'b', label=coord_plot[i] + " torque")
            if coord_plot[i] == 'shoulder_elev':
                if 'shoulder_add' in list(alex_ref.values()):
                    ax.plot(alex_time[alex_time < tf], alex_t_values[alex_time < tf,
                                                                     list(alex_ref.values()).index('shoulder_add')],
                            '--', color='grey', label="shoulder add ref")
            if coord_plot[i] in list(alex_ref.values()):
                ax.plot(alex_time[alex_time < tf], alex_t_values[alex_time < tf,
                                                                 list(alex_ref.values()).index(coord_plot[i])],
                        '--b', label=coord_plot[i] + " ref")
            ax.legend()
            ax.set_ylabel("torque [Nm]", color='b')
            ax.set_xlabel("time [s]")
            plt.title(coord_plot[i])
        plt.tight_layout()
        plt.savefig(save_folder + "torque_plot")

        if len(coord_states) > 2:
            plt.figure("torque plot 2")
            n = len(coord_states) - 2
            for i in range(n):
                ax = plt.subplot(n, 1, i + 1)
                ax.plot(time, torque_states[2+i], 'b', label=coord_plot[2+i] + " torque")
                if coord_plot[2+i] in list(alex_ref.values()):
                    ax.plot(alex_time[alex_time < tf], alex_t_values[alex_time < tf,
                                                                     list(alex_ref.values()).index(coord_plot[2+i])],
                            '--b', label=coord_plot[2+i] + " ref")
                ax.legend()
                ax.set_ylabel("torque [Nm]", color='b')
                ax.set_xlabel("time [s]")
                plt.title(coord_plot[2+i])
            plt.tight_layout()
            plt.savefig(save_folder + "torque_plot_2")

    if show_plot:
        plt.show()
    plt.close('all')


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


def emg_data(emg_file, muscle_file, time_file, ti, tf, unit='sec', factor=1, delay_video=0, plot=False,
             osim_muscle_names=['THE', 'EPB', 'FDSI', 'FPL', 'ECRL', 'PT', 'FCR', 'BRA', 'FDSM', 'BRD', 'EDCM', 'FCU',
                                  'ECU', 'BIClong', 'TRIlat', 'TRIlong', 'DELT1', 'DELT2', 'DELT3', 'PECM1', 'INFSP',
                                  'SUPSP', 'RH', 'TRAPC', 'BICshort']):
    """
    Extract EMG data of a movement of interest
    INPUTS: - emg_file: string, path to EMG data file
            - muscle_file: string, path to EMG muscle names file
            - time_file: string, path to EMG time file
            - ti: float, movement initial time in video
            - tf: float, movement final time in video
            - unit: sting, time unit, default = 'sec'
            - factor: float, optional factor to apply to EMG
            - delay_video: float, delay between the video and EMG data
            - plot: bool, to plot or not the EMG data
            - osim_muscle_names: string list, corresponding osim muscle names
    OUTPUT: - emg_period: dictionary, EMG data of each muscle ({'TRIlong': float array, ...})
            - period: array, [ti, tf] in sec
    """
    emg_lines = open(emg_file, 'r').readlines()
    muscle_names = open(muscle_file, 'r').readline().split()
    time_lines = open(time_file, 'r').readlines()
    emg_data = np.zeros((len(emg_lines), len(muscle_names)))
    emg_time = np.zeros((len(emg_lines), len(muscle_names)))
    for l in range(len(emg_lines)):
        for m in range(len(muscle_names)):
            emg_data[l, m] = float(emg_lines[l].split()[m])*factor
            emg_time[l, m] = float(time_lines[l].split()[m])

    if ti == 0:
        ti = emg_time[0, 0]
        if unit == 'min':
            ti = ti / 60
        if tf != -1 and unit == 'sec':
            tf = ti + tf
    if tf == -1:
        tf = emg_time[-1, 0]
        if unit == 'min':
            tf = tf / 60
    if unit == 'min':
        period = np.array([ti // 1 * 60 + ti % 1 * 100, tf // 1 * 60 + tf % 1 * 100]) - delay_video
    else:
        period = np.array([ti, tf])
    emg_period = {}
    for m in range(len(muscle_names)):
        time_i = emg_time[emg_time[:, m] > period[0], m]
        time = time_i[time_i < period[1]]
        time = time - time[0]
        data = emg_data[emg_time[:, m] > period[0], m]
        data = data[time_i < period[1]]
        emg_period[osim_muscle_names[m]] = [time, data]

    if plot:
        plt.figure()
        for m in range(len(osim_muscle_names)):
            plt.plot(emg_period.get(osim_muscle_names[m])[0], emg_period.get(osim_muscle_names[m])[1] + m*1,
                     label=osim_muscle_names[m])
        plt.legend()
        plt.show()

    return emg_period, period


def alex(alex_torques, values, time, ref, step_size, n_steps):

    torques = {}
    time_sim = np.linspace(0, step_size*n_steps, n_steps)
    for r in alex_torques:
        if r == 'shoulder_add':
            torques[r] = np.interp(time_sim, time, -values[:, list(ref.values()).index(r)])
        else:
            torques[r] = np.interp(time_sim, time, values[:, list(ref.values()).index(r)])

    return torques


def compute_alex_torque(sh_marker, m1, elb_marker, m2, m_arm=2.55, m_farm=1.72):

    g = 9.81
    m1_sh = m1 - sh_marker
    m2_elb = m2 - elb_marker
    m2_sh = m2 - sh_marker
    elb_torque = np.cross(m2_elb, np.array([0, m_farm*g, 0]))
    sh_torque = np.cross(m1_sh, np.array([0, m_arm * g, 0])) + np.cross(m2_sh, np.array([0, m_farm * g, 0]))
    return -sh_torque, -elb_torque


def init_from_alex(osim_file, alex_file, ti, side, unit='sec'):

    if unit == 'min':
        ti = ti // 1 * 60 + ti % 1 * 100

    alex_data = open(alex_file, 'r')
    alex_lines = alex_data.readlines()
    t0 = float(alex_lines[2].split(',')[3])
    t = t0
    l = 2
    while t-t0 < ti:
        l += 1
        t = float(alex_lines[l].split(',')[3])
    if side == 'R':
        sh_flex = alex_lines[l].split(',')[8]
        sh_add = alex_lines[l].split(',')[4]
        sh_rot = alex_lines[l].split(',')[5]
        elb_flex = alex_lines[l].split(',')[9]
        sh_elev, sh_elv, sh_rot = alex_transform_coord(float(sh_flex), float(sh_add), float(sh_rot))
    elif side == 'L':
        sh_flex = alex_lines[l].split(',')[45]
        sh_add = alex_lines[l].split(',')[41]
        sh_rot = alex_lines[l].split(',')[42]
        elb_flex = alex_lines[l].split(',')[46]
        sh_elev, sh_elv, sh_rot = alex_transform_coord(float(sh_flex), float(sh_add), float(sh_rot))
    osim_file = modify_default_Coord(osim_file, 'shoulder_elev', max(min(float(sh_elev), 3.14), 0))
    osim_file = modify_default_Coord(osim_file, 'elv_angle', max(min(float(sh_elv), 2.267), -1.57))
    osim_file = modify_default_Coord(osim_file, 'shoulder_rot', max(min(float(sh_rot), 1.48), -1.48))
    osim_file = modify_default_Coord(osim_file, 'elbow_flexion', max(min(1.57+float(elb_flex), 2.267), 0))
    print('sh_flex: ', sh_flex, 'sh_add: ', sh_add, 'sh_rot: ', sh_rot, 'elb_flex: ', 1.57+float(elb_flex))
    print('sh_flex: ',  sh_elev, 'sh_elev: ', sh_elv, 'sh_rot: ', sh_rot, 'elb_flex: ', 1.57+float(elb_flex))
    return osim_file


def alex_transform_coord(sh_flex, sh_add, sh_rot):

    sh_elv = np.arctan(np.cos(sh_flex)/np.cos(sh_add))  # + sh_rot
    sh_elev = np.arccos(min(max(np.cos(sh_flex)/np.sin(sh_elv), -1), 1))  # (sh_elv-sh_rot)
    return sh_elev, sh_elv, sh_rot


def alex_torque(alex_file, ti=0,
                ref={'R_Torque_3': 'shoulder_elev', 'R_Torque_1': 'shoulder_add', 'R_Torque_2': 'shoulder_rot',
                     'R_Torque_4': 'elbow_flexion'}, plot=False):
    with open(alex_file, 'r') as f1:
        lines = f1.readlines()
        values = np.zeros((len(lines) - 2, len(ref)))
        time = np.zeros(len(lines) - 2)
        joints = lines[0].split(',')
        for r in range(2, len(lines)):
            for j in range(len(joints)):
                if joints[j] in list(ref.keys()):
                    time[r - 2] = float(lines[r].split(',')[3])
                    values[r - 2, list(ref.keys()).index(joints[j])] = float(lines[r].split(',')[j])

    if plot:
        plt.figure()
        for j in range(len(ref)):
            plt.plot(time[time > ti] - time[time > ti][0], values[time > ti, j], label=list(ref.values())[j])
        plt.legend()
        plt.xlabel('time [s]')

    return values[time > ti, :], time[time > ti] - time[time > ti][0], ref


if __name__ == '__main__':
        main()