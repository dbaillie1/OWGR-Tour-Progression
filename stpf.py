# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:27:05 2021

@author: BaillieD
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set();
from math import pi

def produce_tour_progression_sankey(fn):

    df = pd.read_excel(fn, sheet_name="Raw")
    tours = pd.read_excel(fn, sheet_name="Tours")
    players = pd.read_excel(fn, sheet_name="PlayerNames")
    # data
    label = tours['TourName'].to_list()
    
    all_label = label.copy()
    all_label.extend(label)
    all_label.extend(label)
    all_label.extend(label)
    all_label.extend(label)
    all_label.extend(label)
    
    
    
    all_source = np.array((0,))
    all_target = np.array((0,))
    all_value = np.array((0,))
    for y in range(0, 5):
        source = (y * len(tours)) + np.arange(1, len(tours)+1)
        target = (y * len(tours)) + 1 + len(tours) +  np.zeros(np.shape(source))
        source_base = np.arange(1, len(tours)+1)
        target_base =  np.zeros(np.shape(source_base)) + 1
    
        for i in range(2,len(tours)+1):
            source = np.append(source, (y * len(tours)) + np.arange(1, len(tours) + 1))
            target = np.append(target, (y * len(tours)) + len(tours) + i*np.ones(np.shape(np.arange(1, len(tours)+ 1))))
            
            source_base = np.append(source_base, np.arange(1, len(tours) + 1))
            target_base = np.append(target_base,  i*np.ones(np.shape(np.arange(1, len(tours)+ 1))))
        
       
        all_source = np.append(all_source, source)
        all_target = np.append(all_target, target)
        
        
        yr_str1 = str(2015 + y)
        yr_str2 = str(2015 + 1 + y)
        
        value = np.zeros(np.shape(source))
    
        for ii in range(0, len(source_base)):
            booll = (df[yr_str1] == source_base[ii]) & (df[yr_str2] == target_base[ii])
            value[ii] = np.sum(booll)
            
        all_value = np.append(all_value, value)
        
    
    all_source = all_source[1:] - 1
    all_target = all_target[1:] - 1
    all_value = all_value[1:]
    
    # data to dict, dict to sankey
    link = dict(source = all_source, target = all_target, value = all_value)
    node = dict(label = all_label)
    data = go.Sankey(link = link, node=node)
    # plot
    fig = go.Figure(data)
    
    return fig


def produce_tour_progression_parcat(fn):
    df = pd.read_excel(fn, sheet_name='RawStr')
    fig = go.Figure(go.Parcats(
        dimensions=[
            {'label': '2015',
             'values': df['2015'].to_list()},
            {'label': '2016',
             'values': df['2016'].to_list()},
            {'label': '2017',
             'values': df['2017'].to_list()},
            {'label': '2018',
             'values': df['2018'].to_list()},
            {'label': '2019',
             'values': df['2019'].to_list()},
            {'label': '2020',
             'values': df['2020'].to_list()}]
    ))
    
    return fig
