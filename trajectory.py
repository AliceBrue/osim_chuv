import numpy as np
import matplotlib.pyplot as plt
import os
from osim_model import *
from opensim_environment import *

def main():

    traj = False  # to define a new trajectory
    # Inverse Kinematics to perform on OpenSim
    static_optim = True  # to perform Static Optimisation computing muscle activation (or on OpenSim)

    if traj:
        # Define trajectory velocity bell shaped profile
        movement = 'sh_flexion'
        period = 1
        freq = 50
        sh_pos_i = 5
        sh_pos_f = 80
        elb_pos_i = 15
        time = np.linspace(0, period, period*freq)

        v1 = (sh_pos_f - sh_pos_i) * (-4 * 15 * np.power(time, 3) / (period ** 4) +
                                 5 * 6 * np.power(time, 4) / (period ** 5) +
                                 3 * 10 * np.power(time, 2) / (period ** 3))
        v2 = (sh_pos_i - sh_pos_f) * (-4 * 15 * np.power(time, 3) / (period ** 4) +
                                 5 * 6 * np.power(time, 4) / (period ** 5) +
                                 3 * 10 * np.power(time, 2) / (period ** 3))
        v_elb = np.concatenate((np.zeros(period*freq), np.zeros(int(1/2*period*freq)), np.zeros(period*freq)))
        v_sh = np.concatenate((v1, np.zeros(int(1/2*period*freq)), v2))
        p_elb = np.zeros(len(v_elb))
        p_sh = np.zeros(len(v_sh))
        p_elb[0] = elb_pos_i
        p_sh[0] = sh_pos_i
        for t in range(1, len(v_elb)):
            p_elb[t] = np.trapz(v_elb[:t], dx=1/freq) + p_elb[0]
            p_sh[t] = np.trapz(v_sh[:t], dx=1/freq) + p_sh[0]

        # Plot joint kinematics
        time = np.linspace(0, 2.5*period, int(2.5*period*freq))
        plt.figure()
        plt.plot(time, p_elb, color='b', label='elbow angle')
        plt.plot(time, v_elb, color='b', linestyle='dashdot', label='elbow velocity')
        plt.plot(time, p_sh, color='g', label='shoulder angle')
        plt.plot(time, v_sh, color='g', linestyle='dashdot', label='shoulder velocity')
        plt.xlabel('time [s]')
        plt.ylabel('angle [°]')
        plt.title('trajectory')
        plt.legend()

        # Compute corresponding hand marker trajectory
        Xsh = 150*np.ones(len(time))
        Ysh = 460*np.ones(len(time))
        Zsh = 200.0*np.ones(len(time))
        X = Xsh + 300*np.sin(p_sh*3.14/180) + 250*np.sin((p_sh+p_elb)*3.14/180)
        Y = Ysh - 300*np.cos(p_sh*3.14/180) - 250*np.cos((p_sh+p_elb)*3.14/180)
        Z = Zsh

        # Plot hand marker trajectory
        plt.figure()
        plt.plot(X, label='x')
        plt.plot(Y, label='y')
        plt.plot(Z, label='z')
        plt.legend()

        # Write opensim file
        if not os.path.isdir('trajectories/'+movement):
            os.mkdir('trajectories/'+movement)
        file = open('trajectories/'+movement+'/'+movement+'.trc', 'w')
        file.write('PathFileType	4	(X/Y/Z)	traj.trc\n')
        file.write('DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames\n')
        file.write('30.00	30.00\t'+str(len(time))+'\t1	mm	30.00	1\t' +str(len(time))+'\n')
        file.write('Frame#\tTime\twrist\t\t\n')
        file.write('\t\tX1\tY1\tZ1\n')
        file.write('\n')
        for t in range(len(p_elb)):
            file.write(str(t)+'\t'+str(time[t])+'\t'+str(X[t])+'\t'+str(Y[t])+'\t'+str(Z[t])+'\n')

        """file = open('trajectories/'+movement+'/'+movement+'.mot', 'w')
        file.write('Joints\n')
        file.write('version=1\n')
        file.write('nRows='+str(len(time))+'\n')
        file.write('nColumns=4\n')
        file.write('inDegrees=yes\n')
        file.write('endheader\n')
        file.write('time\tshoulder_elev\telbow_flexion\telv_angle\n')
        for t in range(len(p_elb)):
            file.write(str(time[t])+'    '+str(p_sh[t])+' '+str(p_elb[t])+' '+str(89)+'\n')"""

    if static_optim:
        movement = 'sh_flexion'
        osim_file = 'models/full_arm_M_chuv.osim'
        osim_file = modify_default_Coord(osim_file, 'shoulder_elev', 0.1)
        osim_file = modify_default_Coord(osim_file, 'elv_angle', 1.4)
        osim_file = modify_default_Coord(osim_file, 'shoulder_rot', 0.15)
        osim_file = modify_default_Coord(osim_file, 'elbow_flexion', 0.3)
        ik_file = 'trajectories/'+movement+'/IK_'+movement+'.mot'

        reserve_xml = 'trajectories/full_Reserve_Actuators.xml'
        results_dir = 'trajectories/' + movement + '/'
        perform_so(osim_file, ik_file, results_dir, reserve_actuators=reserve_xml)
        comp_so_joints(results_dir + 'SO_Kinematics_q.sto', ik_file, results_dir)
        so_reserve(results_dir + 'SO_StaticOptimization_force.sto', results_dir)
        so_muscle(results_dir + 'SO_StaticOptimization_activation.sto', results_dir)


def perform_so(model_file, ik_file, results_dir, grf_file=None, grf_xml=None, reserve_actuators=None):
    """Perform Static Optimisation using OpenSim.

    Parameters
    ----------
    model_file: str
        OpenSim model (.osim)
    ik_file: str
        kinematics calculated from Inverse Kinematics
    grf_file: str
        the ground reaction forces
    grf_xml: str
        xml description containing how to apply the GRF forces
    reserve_actuators: str
        path to the reserve actuator .xml file
    results_dir: str
        directory to store the results
    """
    # model
    model = OsimModel(model_file, 0.01, 0.0001, body_ext_force=None, visualize=True, save_kin=True,
                      alex_torques=None, moments=None)
    kinematics = opensim.Kinematics()
    kinematics.setModel(model.model)
    model.model.addAnalysis(kinematics)
    #: Initialise osim model
    model.reset()

    # prepare external forces xml file
    if grf_file is not None:
        name = os.path.basename(grf_file)[:-8]
        external_loads = opensim.ExternalLoads(grf_xml, True)
        external_loads.setExternalLoadsModelKinematicsFileName(ik_file)
        external_loads.setDataFileName(grf_file)
        external_loads.setLowpassCutoffFrequencyForLoadKinematics(6)
        external_loads.printToXML(results_dir + name + '.xml')

    # add reserve actuators
    if reserve_actuators is not None:
        force_set = opensim.SetForces(reserve_actuators, True)
        force_set.setMemoryOwner(False)  # model will be the owner
        for i in range(0, force_set.getSize()):
            model.model.updForceSet().append(force_set.get(i))

    # construct static optimization
    motion = opensim.Storage(ik_file)
    static_optimization = opensim.StaticOptimization()
    static_optimization.setStartTime(motion.getFirstTime())
    static_optimization.setEndTime(motion.getLastTime())
    static_optimization.setUseModelForceSet(True)
    static_optimization.setUseMusclePhysiology(True)
    static_optimization.setActivationExponent(2)
    static_optimization.setConvergenceCriterion(0.0001)
    static_optimization.setMaxIterations(100)
    model.model.addAnalysis(static_optimization)

    # analysis
    analysis = opensim.AnalyzeTool(model.model)
    analysis.setName('SO')
    analysis.setModel(model.model)
    analysis.setInitialTime(motion.getFirstTime())
    analysis.setFinalTime(motion.getLastTime())
    analysis.setLowpassCutoffFrequency(6)
    analysis.setCoordinatesFileName(ik_file)
    if grf_file is not None:
        analysis.setExternalLoadsFileName(results_dir + name + '.xml')
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(results_dir)
    analysis.run()
    so_force_file = results_dir + '_so_forces.sto'
    so_activations_file = results_dir + '_so_activations.sto'

    return (so_force_file, so_activations_file)


def comp_so_joints(so_kin_file, IK_file, results_dir,
                   joints=['shoulder_elev', 'elv_angle', 'shoulder_rot', 'elbow_flexion']):
    """Plot SO and IK joints kinematics."""
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

    cmc_data = open(so_kin_file, 'r')
    lines = cmc_data.readlines()
    cmc_joints_i = np.zeros(len(joints))
    coord = lines[10].split()[1:]
    for i in range(len(joints)):
        cmc_joints_i[i] = coord.index(joints[i]) + 1
    cmc_joints_i = cmc_joints_i.astype(int)
    cmc_joints = np.zeros((len(lines[11:]), len(joints)))
    cmc_time = np.zeros(len(lines[11:]))
    for l in range(len(lines[11:])):
        cmc_time[l] = lines[l + 11].split()[0]
        for i in range(len(joints)):
            cmc_joints[l, i] = float(lines[l + 11].split()[cmc_joints_i[i]])

    plt.figure()
    for i in range(len(joints)):
        plt.subplot(len(joints), 1, i + 1)
        plt.plot(IK_time, IK_angles[:, i], label='IK')
        plt.plot(cmc_time, cmc_joints[:, i], label='SO')
        # plt.ylim(-20, 180)
        plt.ylabel('Angle [°]')
        if i == 0:
            plt.legend()
        plt.title(joints[i]+' kinematics')
    plt.xlabel('time [s]')
    plt.tight_layout()
    plt.savefig(results_dir+'SO_joints')


def so_reserve(so_force_file, results_dir):
    """Plot SO reserve actuators."""
    so_data = open(so_force_file, 'r')
    lines = so_data.readlines()
    so_res = lines[14].split()[-4:]
    so_force = np.zeros((len(lines[15:]), len(so_res)))
    so_time = np.zeros(len(lines[15:]))
    for l in range(len(lines[15:])):
        so_time[l] = lines[l + 15].split()[0]
        for i in range(len(so_res)):
            so_force[l, i] = lines[l + 15].split()[-4+i]

    plt.figure()
    for i in range(len(so_res)):
        plt.plot(so_time, so_force[:, i], label=so_res[i])
        plt.legend()
        plt.title('SO reserve actuators')
        plt.xlabel('time [s]')
    plt.savefig(results_dir+'SO_reserve')


def so_muscle(so_acti_file, results_dir, n_musc=10):
    """Plot SO muscle activation."""
    so_data = open(so_acti_file, 'r')
    lines = so_data.readlines()
    so_muscles = lines[8].split()[1:]
    so_acti = np.zeros((len(lines[9:]), len(so_muscles)-4))
    so_time = np.zeros(len(lines[9:]))
    for l in range(len(lines[9:])):
        so_time[l] = lines[l + 9].split()[0]
        for i in range(len(so_muscles)-4):
            so_acti[l, i] = lines[l + 9].split()[i + 1]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    for p in range((len(so_muscles)-4) // n_musc + 1):
        plt.figure()
        for i in range(p * n_musc, min((p + 1) * n_musc, len(so_muscles)-4)):
            plt.plot(so_time, so_acti[:, i], color=colors[i - p * n_musc], label=so_muscles[i])
        plt.legend()
        plt.ylim(0, 1)
        plt.title('SO muscle activation')
        plt.xlabel('time [s]')
        plt.savefig(results_dir+'SO_muscle '+str(p))


if __name__ == '__main__':
        main()
        plt.show()
