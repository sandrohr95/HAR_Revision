from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def down_sampling_wave(df, recently_freq, div_freq_value: int):
    """
    To down sampling the frequency of the dataset

    Example:
        down_sampling_wave(df, 100, 2) --> down sampling from 100Hz to 50Hz
    """

    ms = 1000
    f = ms / recently_freq
    f = str(f) + 'ms'
    datetimeindex = pd.date_range('2018-01-01', periods=df.shape[0], freq=f)
    freq = datetimeindex.freq
    df.index = datetimeindex

    down_sample = df.resample(freq * div_freq_value).asfreq()

    new_freq = recently_freq / div_freq_value

    return down_sample, new_freq


def up_sampling_wave(df, recently_freq, freq_value: int):
    """
       To Up sampling the frequency of the dataset

       Example:
           up_sampling_wave(df, 50, 2) --> up sampling from 50Hz to 100Hz
       """
    ms = 1000
    f = ms / recently_freq
    f = str(f) + 'ms'
    datetimeindex = pd.date_range('2018-01-01', periods=df.shape[0], freq=f)
    freq = datetimeindex.freq
    df.index = datetimeindex

    df_resample = df.resample(freq / freq_value).asfreq()

    fill_value = df_resample[['user-id', 'activity', 'ActivityEncoded']].fillna(method="ffill")

    # Interporlamos para que no queden a Nan los nuevos puntos creados
    interpolate = df_resample[['x-axis', 'y-axis', 'z-axis']].interpolate(method='linear')

    result = pd.concat([fill_value, interpolate], axis=1, sort=False)

    new_freq = recently_freq * freq_value

    return result, new_freq


def visualize_plotly(df, df_resample, recently_freq, new_freq, disp_seconds=3):
    range_disp_actual_freq = int(recently_freq * disp_seconds)
    range_disp_new_freq = int(new_freq * disp_seconds)
    df = df[0:range_disp_actual_freq]
    df_vis = df_resample[0:range_disp_new_freq]

    df_normalize = normalize_data(df)
    df_resample_normalize = normalize_data(df_vis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_resample_normalize.index,
        y=df_resample_normalize['x-axis'],
        name="x-new-freq",
        line_color='blue',
        opacity=0.8))

    fig.add_trace(go.Scatter(
        x=df_normalize.index,
        y=df_normalize['x-axis'],
        name="x-actual-freq",
        line_color='red',
        opacity=0.8))

    ## Use date string to set xaxis range
    fig.update_layout(
        title_text='Frecuencia de muestreo: ' + str(recently_freq) + 'Hz y ' + str(new_freq) + 'Hz durante ' + str(
            disp_seconds) + 'seg')
    plotly.offline.plot(fig, auto_open=True)


def visualize_matplotlib(df, df_resample, recently_freq, new_freq, disp_seconds=3):
    range_disp_actual_freq = int(recently_freq * disp_seconds)
    range_disp_new_freq = int(new_freq * disp_seconds)
    df = df[0:range_disp_actual_freq]
    df_vis = df_resample[0:range_disp_new_freq]

    df_normalize = normalize_data(df)
    df_resample_normalize = normalize_data(df_vis)

    fig, (ax0, ax1) = plt.subplots(nrows=2,
                                   figsize=(15, 10),
                                   sharex=True)
    plot_axis(ax0, df.index, df_normalize[['x-axis', 'y-axis', 'z-axis']], 'X-Axis')
    plot_axis(ax1, df_vis.index, df_resample_normalize['x-axis', 'y-axis', 'z-axis'], 'X-Axis with new frequency')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle('Frecuencia de muestreo: ' + str(recently_freq) + 'Hz y ' + str(new_freq) + 'Hz durante ' + str(
        disp_seconds) + 'seg')
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def normalize_data(df):
    """ Normalización de los datos del acelerómetro"""

    df = df[['x-axis', 'y-axis', 'z-axis']]
    return (df - df.min()) / (df.max() - df.min())


def plot_activity(df, activity):
    X = 'x-axis'
    Y = 'y-axis'
    Z = 'z-axis'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.timestamp,
        y=df[X],
        name=X,
        line_color='blue',
        opacity=0.8))
    fig.add_trace(go.Scatter(
        x=df.timestamp,
        y=df[Y],
        name=Y,
        line_color='red',
        opacity=0.8))
    fig.add_trace(go.Scatter(
        x=df.timestamp,
        y=df[Z],
        name=Z,
        line_color='green',
        opacity=0.8))
    ## Use date string to set xaxis range
    fig.update_layout(title_text='Activity: ' + str(activity))
    plotly.offline.plot(fig, auto_open=True)


def minmaxScaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data[['x-axis', 'y-axis', 'z-axis']])
    data[['x-axis', 'y-axis', 'z-axis']] = scaler.transform(data[['x-axis', 'y-axis', 'z-axis']])
    return data


path_csv = '/home/antonio/Anaconda_Projects/Python_Projects/Human_Activity_Recognition_with_Smartphones/Create_DL_Models/Data_to_work'
df = pd.read_csv(path_csv + '/PAMAP2_Ordered.csv')

recently_freq = 100
freq_value = 5

df = minmaxScaler(df)

""" DOWN SAMPLING """
down_sample, frec_downsampling = down_sampling_wave(df, recently_freq, freq_value)

subset = down_sample[down_sample['Activity'] == 'walking'][1000:2000]
plot_activity(subset, "walking")
subset_df = df[df['Activity'] == 'walking'][5000:5500]
plot_activity(subset_df, "walking")

""" MATPLOTLIB VISUALIZATION """
# visualize_matplotlib(df, down_sample, actual_freq, frec_downsampling)


# down_sample.to_csv(path_csv+'/PAMAP2_df_hand_16g_Ordered_20hz.csv',index=False)


# """ PLOTLY VISUALIZATION """
# visualize_plotly(df, down_sample, actual_freq, frec_downsampling)


# """ UP SAMPLING """

# upsampling, frec_upsampling = up_sampling_wave(df, actual_freq, freq_value)

# """ PLOTLY VISUALIZATION """
# # visualize_plotly(df, upsampling, actual_freq, frec_upsampling)

# """ MATPLOTLIB VISUALIZATION """
# visualize_matplotlib(df, upsampling, actual_freq, frec_upsampling)

""" Save Dataframe """
# upsampling.to_csv(path_csv+'/WISDM_Ordered_100hz.csv',index=False)
