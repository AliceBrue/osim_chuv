#!/usr/bin/env python

import numpy as np
from opensim_environment import OsimModel
import time
import os
import matplotlib.pyplot as plt
from osim_model import *


def main():

    osim_file = 'models/full_arm_M_chuv.osim'
    step_size = 0.01
    integ_acc = 0.0001

    save_folder = 'results/DELT1/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # init joint angles
    osim_file = modify_default_Coord(osim_file, 'elv_angle', 1.2)
    osim_file = modify_default_Coord(osim_file, 'r_shoulder_elev', 0.14)
    osim_file = modify_default_Coord(osim_file, 'shoulder_rot', 0.16)
    osim_file = modify_default_Coord(osim_file, 'elbow_flexion', 0.29)

    #: controls
    sim_period = 5  # in sec
    controls_dict = dict()
    control = 0.2  # in [0, 1]
    period = 2  # in sec
    period = int(period / step_size)
    delay = 1
    delay = int(delay / step_size)
    controls_dict["DELT1"] = control * np.concatenate((np.zeros(delay), np.ones(period)))
    n_steps = int(sim_period/step_size)

    #: plots
    markers = None  #["wrist"]
    coord_plot = ['shoulder_elev', 'elv_angle', 'elbow_flexion']

    osim_control(osim_file, step_size, n_steps, controls_dict, control_delay=delay, integ_acc=integ_acc,
                 markers=markers, coord_plot=coord_plot, save_folder=save_folder, visualize=True,
                 show_plot=True, save_kin=False)


def def_controls(controls_dict, muscle_names, n_steps):
    n_muscles = len(muscle_names)
    controls = np.zeros((n_muscles, n_steps))
    keys = list(controls_dict.keys())
    for k in keys:
        controls[muscle_names.index(k)] = np.concatenate((controls_dict[k], np.zeros(n_steps-len(controls_dict[k]))))
    return controls


def osim_control(osim_file, step_size, n_steps, controls_dict, control_delay=0, integ_acc=0.01,
                 markers=None, coord_plot=None, save_folder="results/", visualize=False, show_plot=False,
                 save_kin=None):

    #: osim model
    model = OsimModel(osim_file, step_size, integ_acc, visualize=visualize, body_ext_force=None, alex_torques=None,
                      save_kin=save_kin, moments=None)
    muscles = model.model.getMuscles()
    n_muscles = muscles.getSize()
    muscle_names = [None]*n_muscles
    for i in range(n_muscles):
        muscle_names[i] = muscles.get(i).getName()

    #: Controls
    controls = def_controls(controls_dict, muscle_names, n_steps)

    #: Initialize
    model.reset()
    model.reset_manager()

    #: States to plot
    if markers is not None:
        marker_pos = np.zeros((len(markers), 3, n_steps - 1))
        marker_vel = np.zeros((len(markers), 3, n_steps - 1))
    if coord_plot is not None:
        coord_states = np.zeros((len(coord_plot), 2, n_steps-1))

    #: Integrate step 0
    #: Actuate the model from control outputs
    model.actuate(np.zeros(n_muscles))
    #: Integration musculoskeletal system
    model.integrate()

    #: Integrate
    mn_rates = np.zeros(n_muscles)
    p = 0
    for j in range(1, n_steps):
        #: Step controller
        for i in range(n_muscles):
                mn_rates[i] = controls[i, j]

        #: Actuate the model from control outputs
        model.actuate(mn_rates)

        #: Integration musculoskeletal system
        model.integrate()

        #: Results from the integration
        res = model.get_state_dict()
        if markers is not None:
            for i in range(len(markers)):
                marker_pos[i, :, j - 1] = res["markers"][markers[i]]["pos"]
                marker_vel[i, :, j - 1] = res["markers"][markers[i]]["vel"]

        if coord_plot is not None:
            for i in range(len(coord_plot)):
                coord_states[i, 0, j-1] = res["coordinate_pos"][coord_plot[i]]
                coord_states[i, 1, j-1] = res["coordinate_vel"][coord_plot[i]]


    #: Markers plot
    time = np.arange(n_steps-1)*step_size
    if markers is not None:
        plt.figure("marker plot Y")
        for i in range(len(markers)):
            ax = plt.subplot(len(markers), 1, i+1)
            lns1 = ax.plot(time, marker_pos[i, 1] * 100, 'b', label="pos_Y")
            ax2 = ax.twinx()
            lns2 = ax2.plot(time, marker_vel[i, 1], 'c', label="vel_Y")
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("pos [cm]", color='b')
            ax2.set_ylabel("vel [m/s]", color='c')
            plt.title(markers[i]+" kinematics")
            plt.savefig(save_folder+"marker_plot_Y")

            plt.figure("marker plot")
            for i in range(len(markers)):
                marker_v = np.sqrt(np.power(marker_vel[i, 0], 2) + np.power(marker_vel[i, 1], 2))
                ax = plt.subplot(len(markers), 1, i + 1)
                lns1 = ax.plot(time, marker_pos[i, 0] * 100, 'b', label="pos_X")
                ax2 = ax.twinx()
                lns2 = ax2.plot(time, marker_vel[i, 0], 'c', label="vel_X")
                lns3 = ax2.plot(time, marker_v, 'r', label="vel")
                lns = lns1 + lns2 + lns3
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs)
                ax.set_xlabel("time [s]")
                ax.set_ylabel("pos [cm]", color='b')
                ax2.set_ylabel("vel [m/s]", color='c')
                plt.title(markers[i] + " kinematics")
                plt.savefig(save_folder + "marker_plot")

    #: Coordinates plot
    if coord_plot is not None:
        plt.figure("coord plot")
        for i in range(len(coord_plot)):
            ax = plt.subplot(len(coord_plot), 1, i + 1)
            ax.plot(time, coord_states[i, 0], 'b', label=coord_plot[i] + " angle", )
            #ax2 = ax.twinx()
            #lns2 = ax2.plot(time, coord_states[i, 1], 'c', label=coord_plot[i] + " vel")
            #lns = lns1 + lns2
            #labs = [l.get_label() for l in lns]
            #ax.legend(lns, labs)
            ax.set_ylabel("angle [rad]", color='b')
            #ax2.set_ylabel("vel [rad/s]", color='c')
            ax.set_xlabel("time [s]")
            plt.title(coord_plot[i] + " kinematics")
        plt.tight_layout()
        plt.savefig(save_folder + "coord_plot")

    if show_plot:
        plt.show()
    plt.close('all')

    #: Save kinematics
    if save_kin:
        model.save_simulation(save_folder)


if __name__ == '__main__':
        main()
