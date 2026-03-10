#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.metrics import mean_squared_error



def plot(df, column="screen_time", title = None):
    plt.figure(figsize=(20, 6))

    if title != None:
        plt.title(title)
        
    plt.plot(df[column],label=column)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.legend()
    plt.show(block=True)

def multiple_plot(data_items, title = None):
    plt.figure(figsize=(20, 6))

    if title != None:
        plt.title(title)
        
    for item in data_items:
        df = item['df']
        column = item['column']
        label = item.get('label', column)
        style = item.get('style', '-')
        plt.plot(df[column], label=label, linestyle=style)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.legend()
    plt.show(block=True)


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) # MAPE
    me = np.mean(forecast - actual) # ME
    mae = np.mean(np.abs(forecast - actual)) # MAE
    mpe = np.mean((forecast - actual)/actual) # MPE
    rmse = np.mean((forecast - actual)**2)**.5 # RMSE
    corr = np.corrcoef(forecast, actual)[0,1] # correlation coeff
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse,
        'corr':corr})

def compute_rmse(forecast, actual):
    return np.sqrt(mean_squared_error(actual, forecast))