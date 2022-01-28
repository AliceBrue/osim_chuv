import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def main():

    sto_file = 'C:/Users/Acer/Documents/BioRob/HBP/NR/SPD002/SCONE/SPD002_Ong_main.sto'  # 'C:/Users/Acer/Documents/BioRob/HBP/NR/MR012/MR012_Ong/healthy_gait_ong_thelen.sto'
    bodies = ['femur_r', 'tibia_r', 'talus_r', 'toes_r']
    ti = 0
    tf = 2
    dt = 0.01
    bodies_joints_dist = {'femur_r-hip_r': 0.15, 'tibia_r-knee_r': 0.15}
    #plot_scone_3D(sto_file, bodies, ti=ti, tf=tf, dt=dt, bodies_joints_dist=bodies_joints_dist)

    var = ['leg0_l.state', 'leg1_r.state']
    tf = 10
    #plot_scone_var(sto_file, var, ti, tf)
    plot_scone_events(sto_file, ti, tf)


def plot_scone_3D(sto_file, bodies, ti=0, tf=-1, dt=1, bodies_joints_dist={}):

    file = open(sto_file, 'r')
    lines = file.readlines()[6:]
    data = np.zeros((len(lines)-1, 3*len(bodies)))
    time = np.zeros(len(lines)-1)
    indexes = np.zeros(3*len(bodies)+len(bodies_joints_dist))
    for b in range(len(bodies)):
        indexes[3*b] = lines[0].split().index(bodies[b]+'.com_pos_x')
        indexes[3*b+1] = lines[0].split().index(bodies[b] + '.com_pos_y')
        indexes[3*b+2] = lines[0].split().index(bodies[b] + '.com_pos_z')
    for j in range(len(bodies_joints_dist)):
        if 'hip' in list(bodies_joints_dist.keys())[j]:
            indexes[3*len(bodies)+j] = lines[0].split().index('hip_flexion_'+list(bodies_joints_dist.keys())[j].split('_')[-1])
        elif 'knee' in list(bodies_joints_dist.keys())[j]:
            indexes[3 * len(bodies) + j] = lines[0].split().index(
                'knee_angle_' + list(bodies_joints_dist.keys())[j].split('_')[-1])
    indexes = indexes.astype(int)
    bodies_joints = []
    for j in range(len(bodies_joints_dist)):
        bodies_joints.append(list(bodies_joints_dist.keys())[j].split('-')[0])
    for r in range(len(lines)-1):
        for b in range(len(bodies)):
            if bodies[b] in bodies_joints:
                j = bodies_joints.index(bodies[b])
                key = list(bodies_joints_dist.keys())[j]
                data[r, 3 * b] = float(lines[r + 1].split()[indexes[3 * b]]) - bodies_joints_dist[key]*np.sin(float(lines[r + 1].split()[indexes[3 * len(bodies)+j]]))
                data[r, 3 * b + 1] = float(lines[r + 1].split()[indexes[3 * b + 1]]) + bodies_joints_dist[key]*np.cos(float(lines[r + 1].split()[indexes[3 * len(bodies)+j]]))
                data[r, 3 * b + 2] = lines[r + 1].split()[indexes[3 * b + 2]]
            else:
                data[r, 3*b] = lines[r+1].split()[indexes[3*b]]
                data[r, 3*b+1] = lines[r+1].split()[indexes[3*b+1]]
                data[r, 3*b+2] = lines[r+1].split()[indexes[3*b+2]]
        time[r] = lines[r+1].split()[0]

    if tf == -1:
        tf = time[-1]
    time_indexes = np.where((time>ti) & (time<tf))[0]

    temp_time = time[time_indexes] - ti
    temp_data = data[time_indexes, :]
    dt = int(dt / (time[1] - time[0]))
    time = np.zeros(len(temp_time)//dt)
    data = np.zeros((len(temp_time) // dt, len(data[0, :])))
    for i in range(len(temp_time)//dt):
        time[i] = temp_time[dt*i]
        data[i] = temp_data[dt * i]

    # 3D trajectory
    #fig = plt.figure()
    #ax = Axes3D(fig)
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('Reds')
    for t in range(len(time)):
        x = np.zeros(len(bodies))
        y = np.zeros(len(bodies))
        z = np.zeros(len(bodies))
        for b in range(len(bodies)):
            x[b] = data[t, 3*b]
            y[b] = data[t, 3*b+1]
            #z[b] = data[t, 3*b+2]
            if t == 0:
                ax.plot(x[b], y[b], c=cmap(0.2+0.8*t/len(time)), marker='o', ms=3 + 2*b, label=bodies[b])
            else:
                ax.plot(x[b], y[b], c=cmap(0.2+0.8*t / len(time)), marker='o', ms=3 + 2*b)
        ax.plot(x, y, c=cmap(0.2+0.8*t / len(time)))
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #ax.set_zlabel('z')
    ax.set_title('Stick plot right leg 1 gait cycle')
    fig.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(vmin=ti, vmax=tf), cmap=cmap), label='time [s]')


def plot_scone_var(sto_file, var, ti=0, tf=-1):

    file = open(sto_file, 'r')
    lines = file.readlines()[6:]
    data = np.zeros((len(lines)-1, len(var)))
    time = np.zeros(len(lines)-1)
    indexes = np.zeros(len(var))
    for v in range(len(var)):
        indexes[v] = lines[0].split().index(var[v])
    indexes = indexes.astype(int)
    for r in range(len(lines)-1):
        for v in range(len(var)):
            data[r, v] = lines[r + 1].split()[indexes[v]]
        time[r] = lines[r+1].split()[0]

    if tf == -1:
        tf = time[-1]
    time_indexes = np.where((time>ti) & (time<tf))[0]

    time = time[time_indexes] - ti
    data = data[time_indexes, :]

    # plot
    fig, ax = plt.subplots()
    for v in range(len(var)):
        ax.plot(time, data[:, v], label=var[v])
    ax.legend()
    ax.set_xlabel('time [s]')
    #ax.set_ylabel('var')
    #ax.set_zlabel('z')
    ax.set_title('Gait variables')


def plot_scone_events(sto_file, ti=0, tf=-1):

    file = open(sto_file, 'r')
    lines = file.readlines()[6:]
    l_state = np.zeros(len(lines)-1)
    r_state = np.zeros(len(lines)-1)
    time = np.zeros(len(lines)-1)
    l_index = lines[0].split().index('leg0_l.state')
    r_index = lines[0].split().index('leg1_r.state')
    for r in range(len(lines)-1):
        l_state[r] = lines[r + 1].split()[l_index]
        r_state[r] = lines[r + 1].split()[r_index]
        time[r] = lines[r+1].split()[0]

    if tf == -1:
        tf = time[-1]
    time_indexes = np.where((time > ti) & (time < tf))[0]
    time = time[time_indexes] - ti
    l_state = l_state[time_indexes]
    r_state = r_state[time_indexes]

    # Foot ON and OFF events
    l_on = np.where((l_state[1:]-l_state[:-1]) < 0)
    l_off = np.where((l_state[1:] - l_state[:-1] > 0) & (l_state[1:] == 3))
    r_on = np.where((r_state[1:] - r_state[:-1]) < 0)
    r_off = np.where((r_state[1:] - r_state[:-1] > 0) & (r_state[1:] == 3))

    # plot
    fig, ax = plt.subplots()
    ax.vlines(l_on, ymin=0, ymax=1, color='b', label='l_on')
    ax.vlines(l_off, ymin=0, ymax=1, color='darkblue', label='l_off')
    ax.vlines(r_on, ymin=0, ymax=1, color='darkorange', label='r_on')
    ax.vlines(r_off, ymin=0, ymax=1, color='red', label='r_off')
    ax.legend()
    ax.set_xlabel('time [s]')
    #ax.set_ylabel('var')
    #ax.set_zlabel('z')
    ax.set_title('Gait variables')


if __name__ == '__main__':
        main()
        plt.show()
