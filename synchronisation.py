import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def main():

    id = 1
    recording = 'test_flexion'
    #kin_file = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/df_cam1_816612061599_record_17_11_2021_1555_35.csv'
    kin_file_2 = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/df_cam1_916512060805_record_17_11_2021_1555_08.csv'
    emg_file = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/emg_norm_data.txt'
    emg_names = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/emg_names.txt'
    emg_time = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/emg_time.txt'
    imu_file = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/imu_data.txt'
    imu_names = 'C:/Users/Acer/Desktop/Pour alice/UP00' + str(id) + '/'+recording+'/imu_names.txt'

    bodies = ['left_wrist', 'left_elbow']
    muscles = ['BBsh', 'FD34']
    ti = 0
    tf = -1

    # kin cam2
    dt = 35-8
    ti = ti + dt
    fproba = True
    kin_data2, kin_time2 = plot_kin(kin_file_2, bodies, ti=ti, tf=tf, fproba=fproba)
    imu_data, imu_time = plot_imu(imu_file, imu_names, muscles)
    #emg_data, emg_time = plot_emg(emg_file, emg_names, emg_time)  # muscles of interest, if None all

    # 3D kin
    plot_kin_3D(kin_file_2, bodies, ti=0, tf=10, fproba=True)

    b = 0
    kin_coord = 1
    m = int(muscles.index('FD34'))
    imu_coord = 1
    kin = kin_data2[:, 4 * b + kin_coord]
    imu = np.interp(kin_time2, imu_time, imu_data[:, 3 * m + imu_coord])
    coord_labels = ['.x', '.y', '.z']
    #cross_correlation(kin, imu, kin_time2, xlabel=bodies[b]+coord_labels[kin_coord],
    #                  ylabel=muscles[m]+coord_labels[kin_coord])


    """# comparison
    fig, axs = plt.subplots(3)
    b=0
    axs[0].plot(kin_time2, kin_data2[:, 4*b], label='x cam 1')
    axs[0].plot(kin_time, kin_data[:, 4*b+1], label='y cam 2')
    axs[0].legend()
    axs[1].plot(kin_time2, kin_data2[:, 4*b+1], label="y cam 1")
    axs[1].plot(kin_time, -kin_data[:, 4*b], label="-x cam 2")
    axs[1].legend()
    axs[2].plot(kin_time2, kin_data2[:, 4*b+2], label="z cam 1")
    axs[2].plot(kin_time, kin_data[:, 4*b+2], label="z cam 2")
    axs[2].legend()
    axs[2].set_xlabel('time [s]')
    plt.suptitle('comparison left wrist tracking')"""


def plot_kin(kin_file, bodies, ti=0, tf=-1, fproba=False):

    with open(kin_file) as csvDataFile:
        # open file as csv file
        rows = list(csv.reader(csvDataFile))
        data = np.zeros((len(rows)-1, 4*len(bodies)))
        time = np.zeros(len(rows)-1)
        indexes = np.zeros(4*len(bodies))
        for b in range(len(bodies)):
            indexes[4*b] = rows[0].index(bodies[b]+'.x')
            indexes[4*b+1] = rows[0].index(bodies[b] + '.y')
            indexes[4*b+2] = rows[0].index(bodies[b] + '.z')
            indexes[4 * b + 3] = rows[0].index(bodies[b] + '.proba')
        indexes = indexes.astype(int)
        for r in range(len(rows)-1):
            for b in range(len(bodies)):
                if fproba and float(rows[r + 1][indexes[4 * b + 3]]) < 0.5:
                    data[r, 4 * b] = np.nan
                    data[r, 4 * b + 1] = np.nan
                    data[r, 4 * b + 2] = np.nan
                    data[r, 4 * b + 3] = rows[r + 1][indexes[4 * b + 3]]
                else:
                    data[r, 4*b] = rows[r+1][indexes[4*b]]
                    data[r, 4*b+1] = rows[r + 1][indexes[4*b+1]]
                    data[r, 4*b+2] = rows[r + 1][indexes[4*b+2]]
                    data[r, 4 * b + 3] = rows[r + 1][indexes[4 * b + 3]]
            time[r] = rows[r+1][-1]

        if tf == -1:
            tf = time[-1]
        time_indexes = np.where((time>ti) & (time<tf))[0]

        if fproba:
            for b in range(len(bodies)):
                data[:, 4*b] = fill_nan(data[:, 4*b])
                data[:, 4 * b+1] = fill_nan(data[:, 4 * b+1])
                data[:, 4 * b+2] = fill_nan(data[:, 4 * b+2])

        plt.figure()
        for b in range(len(bodies)):
            plt.plot(time[time_indexes], data[time_indexes, 4*b] +2*b, label=bodies[b]+'.x')
            plt.plot(time[time_indexes], data[time_indexes, 4*b+1] +2*b, label=bodies[b] + '.y')
            plt.plot(time[time_indexes], data[time_indexes, 4*b+2] +2*b, label=bodies[b] + '.z')
            plt.plot(time[time_indexes], data[time_indexes, 4 * b + 3] +2*b, label=bodies[b] + '.proba')
        plt.legend()
        plt.xlabel('time [s]')
        plt.title('PifPaf tracking')

        return data[time_indexes,:], time[time_indexes]-ti


def plot_kin_3D(kin_file, bodies, ti=0, tf=-1, fproba=False):

    with open(kin_file) as csvDataFile:
        # open file as csv file
        rows = list(csv.reader(csvDataFile))
        data = np.zeros((len(rows)-1, 4*len(bodies)))
        time = np.zeros(len(rows)-1)
        indexes = np.zeros(4*len(bodies))
        for b in range(len(bodies)):
            indexes[4*b] = rows[0].index(bodies[b]+'.x')
            indexes[4*b+1] = rows[0].index(bodies[b] + '.y')
            indexes[4*b+2] = rows[0].index(bodies[b] + '.z')
            indexes[4 * b + 3] = rows[0].index(bodies[b] + '.proba')
        indexes = indexes.astype(int)
        for r in range(len(rows)-1):
            for b in range(len(bodies)):
                if fproba and float(rows[r + 1][indexes[4 * b + 3]]) < 0.5:
                    data[r, 4 * b] = np.nan
                    data[r, 4 * b + 1] = np.nan
                    data[r, 4 * b + 2] = np.nan
                    data[r, 4 * b + 3] = rows[r + 1][indexes[4 * b + 3]]
                else:
                    data[r, 4*b] = rows[r+1][indexes[4*b]]
                    data[r, 4*b+1] = rows[r + 1][indexes[4*b+1]]
                    data[r, 4*b+2] = rows[r + 1][indexes[4*b+2]]
                    data[r, 4 * b + 3] = rows[r + 1][indexes[4 * b + 3]]
            time[r] = rows[r+1][-1]

        if tf == -1:
            tf = time[-1]
        time_indexes = np.where((time>ti) & (time<tf))[0]

        if fproba:
            for b in range(len(bodies)):
                data[:, 4*b] = fill_nan(data[:, 4*b])
                data[:, 4 * b+1] = fill_nan(data[:, 4 * b+1])
                data[:, 4 * b+2] = fill_nan(data[:, 4 * b+2])

        data = data[time_indexes, :]
        time = time[time_indexes] - ti

        # 3D trajectory
        fig = plt.figure()
        ax = Axes3D(fig)
        cmap = cm.get_cmap('Reds')
        for t in range(len(time)):
            x = np.zeros(len(bodies))
            y = np.zeros(len(bodies))
            z = np.zeros(len(bodies))
            for b in range(len(bodies)):
                x[b] = data[t, 4*b]
                y[b] = data[t, 4*b+1]
                z[b] = data[t, 4*b+2]
                if t == 0:
                    ax.plot(x[b], y[b], z[b], c=cmap(1 - 0.8*t/len(time)), marker='o', ms=3 + 2*b, label=bodies[b]+'_i')
                elif t == len(time)-1:
                    ax.plot(x[b], y[b], z[b], c=cmap(1 - 0.8*t / len(time)), marker='o', ms=3 + 2*b, label=bodies[b]+'_f')
                else:
                    ax.plot(x[b], y[b], z[b], c=cmap(1 - 0.8*t / len(time)), marker='o', ms=3 + 2*b)
            ax.plot(x, y, z, c=cmap(1 - t / len(time)))
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('PifPaf tracking')


def plot_imu(imu_file, imu_names, muscles):
    f = open(imu_names, "r")
    names = f.readlines()[0].split()

    f = open(imu_file, "r")
    lines = f.readlines()
    data = np.zeros((len(lines), 3*len(muscles)))
    indexes = np.zeros(3*len(muscles))
    for m in range(len(muscles)):
        indexes[3*m] = 3*names.index(muscles[m])
        indexes[3 * m+1] = 3*names.index(muscles[m]) + 1
        indexes[3 * m + 2] = 3*names.index(muscles[m]) + 2
    indexes = indexes.astype(int)
    for l in range(len(lines)):
        for m in range(3*len(muscles)):
            data[l, m] = float(lines[l].split()[indexes[m]])

    fs = 1.481481481481480e+02
    time = np.arange(0, len(data[:, 0])/fs, 1/fs)

    plt.figure()
    for m in range(len(muscles)):
        plt.plot(time, data[:, 3*m]+2*m, label=muscles[m]+'.x')
        plt.plot(time, data[:, 3*m+1]+2*m, label=muscles[m]+'.y')
        plt.plot(time, data[:, 3*m+2]+2*m, label=muscles[m] + '.z')
    plt.legend()
    plt.xlabel('time [s]')
    plt.title('IMUs')

    return data, time


def plot_emg(emg_file, emg_names, emg_time):

    f = open(emg_file, "r")
    lines = f.readlines()
    data = np.zeros((len(lines), len(lines[0].split())))
    for l in range(len(lines)):
        for m in range(len(lines[0].split())):
            data[l, m] = float(lines[l].split()[m])

    f = open(emg_names, "r")
    names = f.readlines()[0].split()

    f = open(emg_time, "r")
    lines = f.readlines()
    times = np.zeros((len(lines), len(lines[0].split())))
    for l in range(len(lines)):
        for m in range(len(lines[0].split())):
            times[l, m] = float(lines[l].split()[m])

    plt.figure()
    for m in range(len(lines[0].split())):
        plt.plot(times[:, m], data[:, m]+m, label=names[m])
    plt.legend()
    plt.xlabel('time [s]')
    plt.title('EMGs')

    return data, times


def cross_correlation(x, y, time, xlabel='x', ylabel='y'):

    xnorm = (x - np.mean(x)) / np.std(x)
    ynorm = (y - np.mean(y)) / np.std(y)

    plt.figure()
    lags, c, line, ax = plt.xcorr(xnorm, ynorm, maxlags=None)
    plt.axhline(0, color='blue', lw=2)
    lag = np.argmax(abs(c))
    lag = lag-len(time)+1

    plt.figure()
    if lag < 0:
        plt.plot(time, y, label=ylabel)
        plt.plot(time[-lag:] - time[-lag] + time[0], x[-lag:], label=xlabel)
    else:
        plt.plot(time[lag:] - time[lag] + time[0], y[lag:], label=ylabel)
        plt.plot(time, x, label=xlabel)
    plt.legend()
    plt.xlabel('time [s]')
    plt.title('sync signals')


def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B


if __name__ == '__main__':
        main()
        plt.show()