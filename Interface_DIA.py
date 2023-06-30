# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QListWidget,QListWidgetItem,QMessageBox
from PyQt5.QtGui import QPixmap 
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from datetime import datetime            
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import preprocessing
from sklearn import neighbors
import sys
import os, os.path
import re
import statistics


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_col_names(PATH):
    xls = pd.ExcelFile(PATH[0])
    sheetX = xls.parse(0)

    liste_frequences=sheetX['frequence']
    
    liste_frequences_for_title=[]
    for i in range(len(liste_frequences)):
        liste_frequences_for_title.append('frequence '+str(liste_frequences[i])+' Hz')

    liste_temps=sheetX['temps']
    liste_temps_for_title=[]
    for i in range(len(liste_temps)):
        liste_temps_for_title.append('temps '+str(liste_temps[i])+' s')

    col_r_m= ['r_m ' + s for s in liste_frequences_for_title]
    col_r_p= ['r_p ' + s for s in liste_frequences_for_title]
    col_t_m= ['t_m ' + s for s in liste_frequences_for_title]
    col_t_p= ['t_p ' + s for s in liste_frequences_for_title]
    col_r= ['r ' + s for s in liste_temps_for_title]
    col_t= ['t ' + s for s in liste_temps_for_title]

    col_names=['Date']+col_r_m+col_r_p+col_t_m+col_t_p+col_r+col_t
    return col_names
    
# def extract_data(PATH)
    # final_list1 = []
    
    # for path in PATH:
            # xls = pd.ExcelFile(path)
            # sheet_names=xls.sheet_names
            
            # for i in range(len(sheet_names)):
                # sheetX = xls.parse(i)
                # liste_frequence=sheetX['frequence'].tolist()
                # liste_reflection_module=sheetX['Réflection modules'].tolist()
                # liste_reflection_phase=sheetX['Réflection phases'].tolist()
                # liste_transmission_module=sheetX['Transmission modules'].tolist()
                # liste_transmission_phase=sheetX['Transmission phases'].tolist()
                # liste_temps=sheetX['temps'].tolist()
                # liste_reflection=sheetX['Réflection'].tolist()
                # liste_transmission=sheetX['Transmission'].tolist()
                
                # final_list1.append(liste_frequence)
                # final_list1.append(liste_reflection_module)
                # final_list1.append(liste_reflection_phase)
                # final_list1.append(liste_transmission_module)
                # final_list1.append(liste_transmission_phase)
                # final_list1.append(liste_temps)
                # final_list1.append(liste_reflection)
                # final_list1.append(liste_transmission)
                
                # print(str(liste_frequence))
                
    # return final_list1

def read_xls(PATH):                       #lire le fichier excel 
        final_list=[]
        data_list=[]
            
        for path in PATH:
            xls = pd.ExcelFile(path)
            sheet_names=xls.sheet_names
            _, ext = os.path.splitext(path)
            date_str=[]
            for str_ in sheet_names:
                date_str.append(str_[7:])
            date_time_obj=[]
            if ext.lower() == '.xls':
                for i in range(len(date_str)):
                    dt_obj = datetime.strptime(date_str[i][:15],"%Y%m%d_%H%M%S")
                    date_time_obj.append(dt_obj)
                    
            elif ext.lower() == '.xlsx':
                for i in range(len(date_str)):
                    dt_obj = datetime.strptime(date_str[i],"%Y%m%d_%H%M%S ")
                    date_time_obj.append(dt_obj)

            dates=[]
            for i in range(len(date_time_obj)):
                date_time = date_time_obj[i].strftime("%Y-%m-%d %H:%M:%S")
                dates.append(date_time)

            for i in range(len(sheet_names)):
                sheetX = xls.parse(i)
                liste_frequence=sheetX['frequence'].tolist()
                liste_reflection_module=sheetX['Réflection modules'].tolist()
                #print("liste_reflection_module",liste_reflection_module)
                liste_reflection_phase=sheetX['Réflection phases'].tolist()
                liste_transmission_module=sheetX['Transmission modules'].tolist()
                liste_transmission_phase=sheetX['Transmission phases'].tolist()
                liste_temps=sheetX['temps'].tolist()
                liste_reflection=sheetX['Réflection'].tolist()
                liste_transmission=sheetX['Transmission'].tolist()
                
                data_list.append(liste_frequence)
                data_list.append(liste_reflection_module)
                data_list.append(liste_reflection_phase)
                data_list.append(liste_transmission_module)
                data_list.append(liste_transmission_phase)
                data_list.append(liste_temps)
                data_list.append(liste_reflection)
                data_list.append(liste_transmission)
                
                final_list.append([dates[i]]+liste_reflection_module+liste_reflection_phase+liste_transmission_module+liste_transmission_phase+liste_reflection+liste_transmission)

        print("len(data_list)",len(data_list))
        
        col_names=get_col_names(PATH)
        df_final=pd.DataFrame(data=final_list,columns=col_names)
        df_final.set_index('Date',inplace=True)
        df_final.sort_index(inplace=True)
        
        
        return df_final,data_list


def get_freq(df_r_m):
    freq1=[]
    for str_ in df_r_m.index:
        freq1.append(str_.replace('r_m frequence ', ''))
    freq2=[]
    for str_ in freq1:
        freq2.append(str_.replace(' Hz', ''))
    freq=[float(i)*1e-9 for i in freq2]
    return freq

def adapt_form_df(df,ech,freq):
    new_df=df[df.columns[ech]].copy()
    new_df.index=freq
    new_df=new_df.to_frame()
    new_df.reset_index(inplace=True)
    new_df.columns=['Freq [GHz]',ech]
    return new_df

def surface(data,start_freq,end_freq):
  tmp1=data[data['Freq [GHz]']<end_freq]
  tmp2=tmp1[tmp1['Freq [GHz]']>start_freq]
  surf=abs(tmp2).sum(axis=0).iloc[1]
  return surf

def minimum(data,start_freq,end_freq):
  tmp1=data[data['Freq [GHz]']<end_freq]
  tmp2=tmp1[tmp1['Freq [GHz]']>start_freq]
  min_=tmp2.min(axis=0).iloc[1]
  freq =tmp2.loc[tmp2[tmp2.columns[1]] == min_].values[0][0]
  return freq,min_

def derivee_vals(data,bandes_frequences):
  max_=[]
  min_=[]
  for i in range(len(bandes_frequences)-1):
    tmp = data

    tmp1=tmp[tmp['Freq [GHz]']<bandes_frequences[i+1]]
    tmp2=tmp1[tmp1['Freq [GHz]']>bandes_frequences[i]]

    tmp3=tmp2[tmp2.columns[1]].diff()
    max_.append(tmp3.max())
    min_.append(tmp3.min())

  return min_,max_

def valeur_a_3_GHz_Liss(data,window):
    return data.rolling(window=window,center=False).mean().iloc[-1].iloc[1]

def valeur_a_1_5_GHz_Liss(t_m,window):
    t_m["Freq [GHz] ro"]=t_m['Freq [GHz]'].round(2)
    t_m["liss"]=t_m[t_m.columns[1]].rolling(window=window,center=False).mean()
    return t_m.loc[t_m["Freq [GHz] ro"]==1.50].iloc[1]["liss"]

def predict_eps(diametre,liste_metriques,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain,data_list):
    # print("data_list[0]",data_list[0])
    # print("data_list[1]",data_list[1])
    # print("df_r_m",df_r_m)
    freq=get_freq(df_r_m)
    #print("freq=",freq)
    PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))

    PATH_MET_DIAM=PATH_WHERE_EXECUTED+'/METRIQUES/metriques_diam_'+str(diametre)+'.csv'
    df_met_diam=pd.read_csv(PATH_MET_DIAM)

    df_train_test_diam=df_met_diam[["eps'","eps''","bande freq 1"]+liste_metriques]
    df_train_test_eps1=df_train_test_diam

    Y_train_eps1=df_train_test_eps1["eps'"]
    
    X_train_1=df_train_test_eps1.drop(["eps'", "eps''"], axis=1)
    scaler_1 = preprocessing.StandardScaler().fit(X_train_1)
    X_train_1=scaler_1.transform(X_train_1)
    
    

    model_eps1 = neighbors.KNeighborsRegressor(n_neighbors = nbr_voisins)
    model_eps1.fit(X_train_1, Y_train_eps1)
    
    eps1_pred=[]
    for ech in range(0,df_terrain.shape[0]): # 0 à 70
        freq_list = data_list[8*ech] # 1ere ligne de data_list (fréquences)
        r_m_list = data_list[8*ech+1] # 2eme ligne de data_list (r_m)


        r_m=adapt_form_df(df_r_m,ech,freq)
        ###### trouver bf1
        x=df_r_m[df_r_m.columns[ech]].rolling(window=15,center=True).mean()
        peaks, _ = find_peaks(x,distance=50)
        f=x[peaks].index[1].replace('r_m frequence ','')
        bf1=float(f.replace(' Hz',''))*1e-9

        #f1 : between 0.4 et 0.9
        I_min1 = [i for i in range(len(freq_list)) if freq_list[i]>=0.4*1e9]
        index_min1 = I_min1[0]
        f_min1 = freq_list[I_min1[0]]
        I_max1 = [i for i in range(len(freq_list)) if freq_list[i]<=0.9*1e9]
        index_max1 = I_max1[-1]
        f_max1 = freq_list[I_max1[-1]]
        r_m_sublist1 = r_m_list[index_min1:index_max1+1]
        MIN1 = min(r_m_sublist1)
        index_freq_1 = [i for i in range(len(r_m_list)) if r_m_list[i] == MIN1]
        freq_1 = freq_list[index_freq_1[0]]
        freq_res1 = float(freq_1)*1e-9
        
        #f2 : between 1.0 et 1.5
        I_min2 = [i for i in range(len(freq_list)) if freq_list[i]>=1.0*1e9]
        index_min2 = I_min2[0]
        f_min2 = freq_list[I_min2[0]]
        I_max2 = [i for i in range(len(freq_list)) if freq_list[i]<=1.5*1e9]
        index_max2 = I_max2[-1]
        f_max2 = freq_list[I_max1[-1]]
        r_m_sublist2 = r_m_list[index_min2:index_max2+1]
        MIN2 = min(r_m_sublist2)
        index_freq_2 = [i for i in range(len(r_m_list)) if r_m_list[i] == MIN2]
        freq_2 = freq_list[index_freq_2[0]]
        freq_res2 = float(freq_2)*1e-9
        
        #f3 : between 2.25 et 2.35
        I_min3 = [i for i in range(len(freq_list)) if freq_list[i]>=2.25*1e9]
        index_min3 = I_min3[0]
        f_min3 = freq_list[I_min3[0]]
        I_max3 = [i for i in range(len(freq_list)) if freq_list[i]<=2.35*1e9]
        index_max3 = I_max3[-1]
        f_max3 = freq_list[I_max3[-1]]
        r_m_sublist3 = r_m_list[index_min3:index_max3+1]
        MIN3 = min(r_m_sublist3)
        index_freq_3 = [i for i in range(len(r_m_list)) if r_m_list[i] == MIN3]
        freq_3 = freq_list[index_freq_3[0]]
        freq_res3 = float(freq_3)*1e-9
        
    # r_m surface bande_freq 1
        surf = surface(r_m[['Freq [GHz]',ech]],0.1,bf1)
    
    # r_m minimum bande_freq 1
        r_m_min = minimum(r_m,0.1,freq_res1)[1]
    
    # r_m minimum derivee bande_freq 1
        deriv_min = derivee_vals(r_m,[0.1,bf1])[0][0]
    
    # r_m maximum derivee bande_freq 1
        deriv_max = derivee_vals(r_m,[0.1,bf1])[1][0]
        
        liste_metriques_eps1=[0.1]
        if('bande freq 2' in liste_metriques):
            liste_metriques_eps1.append(bf1)
        if('r_m surface bande_freq 1' in liste_metriques):   #surface
            liste_metriques_eps1.append(surf)
        if('r_m minimum bande_freq 1' in liste_metriques):  #amp min 
            liste_metriques_eps1.append(r_m_min)
        if('r_m freq minimum bande_freq 1' in liste_metriques):
            liste_metriques_eps1.append(freq_res1)
        if('r_m freq minimum bande_freq 2' in liste_metriques):
            liste_metriques_eps1.append(freq_res2)
        if('r_m freq minimum bande_freq 3' in liste_metriques):
            liste_metriques_eps1.append(freq_res3)
        
        if('r_m minimum derivee bande_freq 1' in liste_metriques):
            liste_metriques_eps1.append(deriv_min)
        if('r_m maximum derivee bande_freq 1' in liste_metriques):
            liste_metriques_eps1.append(deriv_max)
            #print("derivee_vals(r_m,[0.1,bf1])[0][0]",derivee_vals(r_m,[0.1,bf1])[0][0])
    
        print("liste_metriques_eps1",liste_metriques_eps1)
        pred_eps1=model_eps1.predict(scaler_1.transform([liste_metriques_eps1]))
        eps1_pred.append(pred_eps1[0])
        
    return eps1_pred

def predict_eps2(diametre,metriques_2,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain):
    freq=get_freq(df_r_m)

    PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
    PATH_MET_DIAM=PATH_WHERE_EXECUTED+'/METRIQUES/metriques_diam_'+str(diametre)+'.csv'
    df_met_diam=pd.read_csv(PATH_MET_DIAM)

    df_train_test_diam=df_met_diam[["eps'","eps''"]+metriques_2]
    df_train_test_eps2=df_train_test_diam

    Y_train_eps2=df_train_test_eps2["eps''"]

    X_train_2=df_train_test_eps2.drop(["eps'", "eps''"], axis=1)
    scaler_2 = preprocessing.StandardScaler().fit(X_train_2)
    X_train_2=scaler_2.transform(X_train_2)

    model_eps2 = neighbors.KNeighborsRegressor(n_neighbors = nbr_voisins)
    model_eps2.fit(X_train_2, Y_train_eps2)    

    t_m_mesure=[]
    for ech in range(0,df_terrain.shape[0]):
        # MESURES
        t_m = adapt_form_df(df_t_m,ech,freq)
    
    # t_m valeur a 0.4 GHz
        Y_train_0_4 = df_train_test_eps2["t_m valeur a 0.4 GHz"]
        f_mesure = t_m['Freq [GHz]'].iloc[(t_m['Freq [GHz]'] - 0.394).abs().idxmin()]
        t_m_mesure.append(t_m.loc[t_m['Freq [GHz]'] == f_mesure, ech].iloc[0])
        
    eps2_pred=[]
    for ech in range(0,df_terrain.shape[0]):
        moy = abs(Y_train_0_4.mean() - statistics.mean(t_m_mesure))
        t_m_proche = Y_train_0_4.loc[(Y_train_0_4 - t_m_mesure[ech] - moy).abs().idxmin()]
        eps2_proche = Y_train_eps2.loc[(Y_train_0_4 - t_m_mesure[ech] - moy).abs().idxmin()]

        liste_metriques_eps2=[]
        if('t_m valeur a 0.4 GHz' in metriques_2):
            eps2_pred.append(eps2_proche)
        
    return eps2_pred

def get_liste_met():

    PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))

    DIR = PATH_WHERE_EXECUTED+'/METRIQUES'
    L=[name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    MET1=[]
    for str_ in L:
        MET1.append(str_.replace('metriques_diam_', ''))
    MET2=[]
    for str_ in MET1:
        MET2.append(str_.replace('.csv', ''))
    return MET2

#----------------METRIQUES----------------------

# def find_bandes_frequences(data,distance=90):
#   x=data
#   #peaks, _ = find_peaks(x[x.columns[1]],distance=distance)
#   #freq_=x.iloc[peaks]['Freq [GHz]'].tolist()
#   #print("Yes:")
#   
#   freq_ = []
#   
#   freq_list = x[x.columns[0]]
#   db_list = x[x.columns[1]]
#   
#   #frequencies between 0.4 and 0.9
#   Fexpvide1 = 0.610
#   Fnumvide1 = 0.585
#   
#   index_min1 = freq_list[freq_list.gt(0.4)].index[0]
#   f_min1 = freq_list.get(key = index_min1)
#   
#   index_max1 = freq_list[freq_list.le(0.9)].index[-1]
#   f_max1 = freq_list.get(key = index_max1)
#   
#   db_sublist1 = db_list.loc[index_min1:index_max1]
#   
#   #print(str(db_sublist.min()))
#   index_freq_1 = db_sublist1[db_sublist1 == db_sublist1.min()].index[0]
#   freq_1 = freq_list.iloc[index_freq_1]
#   freq_1 = Fexpvide1 - Fnumvide1 + freq_1
#   #print("<---->")
#   #print(str(f_min) + ":" + str(f_max))
#   
#   freq_.append(freq_1)
#   
#   #frequencies between 1.0 and 1.5
#   Fexpvide2 = 1.691
#   Fnumvide2 = 1.440
#   index_min2 = freq_list[freq_list.gt(1.0)].index[0]
#   f_min2 = freq_list.get(key = index_min2)
#   
#   index_max2 = freq_list[freq_list.le(1.5)].index[-1]
#   f_max2 = freq_list.get(key = index_max2)
#   
#   db_sublist2 = db_list.loc[index_min2:index_max2]
#   
#   
#   index_freq_2 = db_sublist2[db_sublist2 == db_sublist2.min()].index[0]
#   freq_2 = freq_list.iloc[index_freq_2]
#   freq_2 = Fexpvide2 - Fnumvide2 + freq_2
#   
#   freq_.append(freq_2)
#   
#   
#   #frequencies between 2.25 and 2.35
#   Fexpvide3 = 2.480
#   Fnumvide3 = 2.388
#   index_min3 = freq_list[freq_list.gt(2.25)].index[0]
#   f_min3 = freq_list.get(key = index_min3)
#   
#   index_max3 = freq_list[freq_list.le(2.35)].index[-1]
#   f_max3 = freq_list.get(key = index_max3)
#   
#   db_sublist3 = db_list.loc[index_min3:index_max3]
#   
#   
#   index_freq_3 = db_sublist3[db_sublist3 == db_sublist3.min()].index[0]
#   freq_3 = freq_list.iloc[index_freq_3]
#   freq_3 = Fexpvide3 - Fnumvide3 + freq_3
#   freq_.append(freq_3)
#   #print(str(freq_3))
#   
#   
#   #print("No:" + str(freq_))
#   
#   freq=[]
#   if len(freq_)==4:
#     freq=[0.1]+freq_
#   if len(freq_)==3:
#     freq=[0.1]+freq_+[3]
#   if len(freq_)==2:
#     freq=[0.1]+freq_+[2.9,3]
#   if len(freq_)==1:
#     freq=[0.1]+freq_+[2.8,2.9,3]
#   
#   return freq
def find_bandes_frequences(data,distance=90):
  x=data
  peaks, _ = find_peaks(x[x.columns[1]],distance=distance)
  freq_=x.iloc[peaks]['Freq [GHz]'].tolist()
  freq=[]
  if len(freq_)==4:
    freq=[0.1]+freq_
  if len(freq_)==3:
    freq=[0.1]+freq_+[3]
  if len(freq_)==2:
    freq=[0.1]+freq_+[2.9,3]
  if len(freq_)==1:
    freq=[0.1]+freq_+[2.8,2.9,3]
  
  return freq
def periodicite(data,seuil):
  ligne = np.ones(len(data))*seuil
  idx = np.argwhere(np.diff(np.sign(data - ligne))).flatten()
  return len(idx)/2

def freq_montee(data,nb_freq):

  data_diff=data.iloc[:,1].diff()
  peaks, _ = find_peaks(data_diff,threshold=100)
  freq_=data.iloc[peaks]['Freq [GHz]'].tolist()
  return freq_[0:nb_freq]

def freq_premiere_montee(data,start_freq,end_freq):
  tmp1=data[data['Freq [GHz]']<end_freq]
  tmp2=tmp1[tmp1['Freq [GHz]']>start_freq]
  tmp3=tmp2.iloc[:,1].diff()
  tmp4=tmp3[tmp3>100] #100 premier seuil de dépassement
  freq=tmp2.loc[tmp4.first_valid_index()][0]
  return freq

def valeur_a_X_GHz(data,X_GHz):
  return data.loc[data["Freq [GHz]"] == X_GHz, data.columns.to_list()[1]].iloc[0]

def get_eps(data,nb_eps):
  eps1=[]
  eps2=[]
  tmp=data.columns.to_list()[1:nb_eps]
  for i in range(len(tmp)):
    l=re.findall(r'[-+]?\d*\.\d+|\d+', tmp[i])
    eps1.append(l[3])
    eps2.append(l[5])
  return eps1,eps2

def extractionnsaveMet(nb_eps,diam,df_S11,df_S21):
    final_list=[]

    eps1=get_eps(df_S21,nb_eps)[0]
    eps2=get_eps(df_S21,nb_eps)[1]

    col_names_S11=df_S11.columns.to_list()
    col_names_S21=df_S21.columns.to_list()

    for i in range(len(eps1)):
        bandes_frequences=find_bandes_frequences(df_S11[["Freq [GHz]",col_names_S11[i+1]]])
        min_,max_=derivee_vals(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences)
        frequences_montee=freq_montee(df_S11[["Freq [GHz]",col_names_S11[i+nb_eps]]],3)
                    
        tmp_list=[eps1[i],
                    eps2[i],
                    "BOIS",
                    diam,

                    bandes_frequences[0],
                    bandes_frequences[1],
                    bandes_frequences[2],
                    bandes_frequences[3],
                    bandes_frequences[4],

                    surface(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[0],bandes_frequences[1]),
                    surface(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[1],bandes_frequences[2]),
                    surface(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[2],bandes_frequences[3]),
                    surface(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[3],bandes_frequences[4]),
                    
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[0],bandes_frequences[1])[1],
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[1],bandes_frequences[2])[1],
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[2],bandes_frequences[3])[1],
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[3],bandes_frequences[4])[1],

                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[0],bandes_frequences[1])[0],
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[1],bandes_frequences[2])[0],
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[2],bandes_frequences[3])[0],
                    minimum(df_S11[["Freq [GHz]",col_names_S11[i+1]]],bandes_frequences[3],bandes_frequences[4])[0],

                    min_[0],
                    min_[1],
                    min_[2],
                    min_[3],

                    max_[0],
                    max_[1],
                    max_[2],
                    max_[3],

                    #frequences_montee[0],
                    #frequences_montee[1],
                    #frequences_montee[2],
                
                    #valeur_a_X_GHz(df_S21[["Freq [GHz]",col_names_S21[i+1]]],df_S11.iloc[(df_S11['Freq [GHz]']-1.5).abs().argsort()[:1]]["Freq [GHz]"].iloc[0]),
                    #valeur_a_X_GHz(df_S21[["Freq [GHz]",col_names_S21[i+1]]],df_S11.iloc[(df_S11['Freq [GHz]']-3.0).abs().argsort()[:1]]["Freq [GHz]"].iloc[0]),
                    
                    #freq_premiere_montee(df_S21[["Freq [GHz]",col_names_S21[i+nb_eps]]],0.1,1),

                    periodicite(df_S21[col_names_S21[i+nb_eps]],150),

                    valeur_a_X_GHz(df_S21[["Freq [GHz]",col_names_S21[i+1]]],df_S11.iloc[(df_S11['Freq [GHz]']-0.394).abs().argsort()[:1]]["Freq [GHz]"].iloc[0])
                    
                    ]
        final_list.append(tmp_list)


    col_names_f_df=["eps'",
                    "eps''",
                    "Type",

                    "Diametre",
                    

                    "bande freq 1",
                    "bande freq 2",
                    "bande freq 3",
                    "bande freq 4",
                    "bande freq 5",

                    "r_m surface bande_freq 1",
                    "r_m surface bande_freq 2",
                    "r_m surface bande_freq 3",
                    "r_m surface bande_freq 4",

                    "r_m minimum bande_freq 1",
                    "r_m minimum bande_freq 2",
                    "r_m minimum bande_freq 3",
                    "r_m minimum bande_freq 4",

                    "r_m freq minimum bande_freq 1",
                    "r_m freq minimum bande_freq 2",
                    "r_m freq minimum bande_freq 3",
                    "r_m freq minimum bande_freq 4",

                    "r_m minimum derivee bande_freq 1",
                    "r_m minimum derivee bande_freq 2",
                    "r_m minimum derivee bande_freq 3",
                    "r_m minimum derivee bande_freq 4",

                    "r_m maximum derivee bande_freq 1",
                    "r_m maximum derivee bande_freq 2",
                    "r_m maximum derivee bande_freq 3",
                    "r_m maximum derivee bande_freq 4",

                    #"r_p freq montee bande_freq 1",
                    #"r_p freq montee bande_freq 2",
                    #"r_p freq montee bande_freq 3",
                    
                    #"t_m valeur a 1.5 GHz",
                    #"t_m valeur a 3 GHz",

                    #"t_p freq premiere montee",
                    
                    "t_p periodicite",
                    "t_m valeur a 0.4 GHz"
                    ]
    final_df=pd.DataFrame(data=final_list,columns=col_names_f_df)
    
    PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))

    path_to_save=PATH_WHERE_EXECUTED+'/METRIQUES/metriques_diam_'+str(diam)+'.csv'
    final_df.to_csv(path_to_save,index = False)

#----------------METRIQUES----------------------


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 500)
        MainWindow.setMinimumSize(QtCore.QSize(700, 500))
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")



        ######### TAB 0 #########

        self.tab_0 = QtWidgets.QWidget()
        self.tab_0.setObjectName("tab_0")
        
        self.label_0 = QtWidgets.QLabel(self.tab_0)
        self.label_0.setObjectName("label_0")
        
        PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
        image_path=PATH_WHERE_EXECUTED+'\couleur.png'
        qpixmap = QPixmap(image_path)
        self.lbl_img = QtWidgets.QLabel(self.tab_0)        
        
        self.comboBox_0 = QtWidgets.QComboBox(self.tab_0)
        self.comboBox_0.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_0.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_0.setObjectName("comboBox_0")
        self.comboBox_0.addItem("")
        self.comboBox_0.addItem("")
        
        
        self.pushButton_0 = QtWidgets.QPushButton(self.tab_0)
        #suivant
        self.pushButton_0.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_0.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
)
        self.pushButton_0.setObjectName("pushButton_0")

        

        ######### TAB 1 #########



        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab_0, "")
        self.listWidget = QtWidgets.QListWidget(self.tab)
        #fenetre aff fichiers
        self.listWidget.setObjectName("listWidget")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab)
        #bouton parcourir
        font = QtGui.QFont()
        font.setPointSize(1)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_7.setStyleSheet("QPushButton{\n"
"    background-color: darkblue;\n"
"    color: rgb(255, 255, 255);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: beige;\n"
"    font: bold 24px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(41, 128, 185);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: beige;\n"
"    font: bold 24px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: darkblue;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: beige;\n"
"    font: bold 24px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_7.setObjectName("pushButton_7")
        self.comboBox_8 = QtWidgets.QComboBox(self.tab)
        #selction ref trans
        self.comboBox_8.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_8.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_8.setObjectName("comboBox_8")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_9 = QtWidgets.QComboBox(self.tab)
        #module phase
        self.comboBox_9.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_9.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_9.setObjectName("comboBox_9")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        
        self.pushButton_27 = QtWidgets.QPushButton(self.tab)
        #supprimer
        self.pushButton_27.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_27.setStyleSheet("    background-color: rgb(191, 191, 191);\n"
    "    border-style: outset;\n"
    "    border-width: 2px;\n"
    "    border-radius: 5px;\n"
    "    border-color: beige;\n"
    "    font: 20px;\n"
    "    min-width: 7em;\n"
    "    padding: 2px;")
        self.pushButton_27.setObjectName("pushButton_27")
        #déselectionner
        self.pushButton_25 = QtWidgets.QPushButton(self.tab)
        self.pushButton_25.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_25.setStyleSheet("QPushButton{\n"
"    background-color: rgb(191, 191, 191);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 5px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: 20px;\n"
"    min-width: 7em;\n"
"    padding: 2px;}")
        self.pushButton_25.setObjectName("pushButton_25")
       #tout selectionner
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("    background-color: rgb(191, 191, 191);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 5px;\n"
"    border-color: beige;\n"
"    font: 20px;\n"
"    min-width: 7em;\n"
"    padding: 2px;")
        self.pushButton_2.setObjectName("pushButton_2")
        
        self.pushButton_8 = QtWidgets.QPushButton(self.tab)
        #afficher
        self.pushButton_8.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_8.setStyleSheet("QPushButton{\n"
"    background-color: rgb(191, 191, 191);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 5px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 7em;\n"
"    padding: 2px;}"
"QPushButton:hover{\n"
"    background-color: rgb(242, 243, 244);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 5px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 7em;\n"
"    padding: 2px;}"
"QPushButton:pressed{"
"    background-color: rgb(191, 191, 191);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 5px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 7em;\n"
"    padding: 2px;}")
        self.pushButton_8.setObjectName("pushButton_8")
        self.graphicsView = PlotWidget(self.tab)
 #("    fenêtre Affichage mesures ")       
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        #suivant
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
)
        self.pushButton.setObjectName("pushButton")


        ######### TAB 2 #########


        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label = QtWidgets.QLabel(self.tab_2)
         # Choix BD numérique
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.tab_2)
        self.comboBox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox.setObjectName("comboBox")

        PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))

        DIR = PATH_WHERE_EXECUTED+'/METRIQUES'
        for i in range(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])):
            self.comboBox.addItem("")
        
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab_2)
         #checkbox choix des métriques
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_10 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_10.setObjectName("checkBox_10")
        self.checkBox_11 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_11.setObjectName("checkBox_11")
        self.checkBox_3 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_7 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_7.setObjectName("checkBox_7")
        self.checkBox_8 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_8.setObjectName("checkBox_8")
        self.checkBox_8.setVisible(False)
        self.checkBox_12 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_12.setObjectName("checkBox_12")
        self.checkBox_12.setVisible(False)
        
        self.comboBox.currentIndexChanged.connect(self.handleComboBoxChange)
        self.comboBox.setCurrentIndex(-1)


        self.label_3 = QtWidgets.QLabel(self.tab_2)
         #nombre de voisins en texte
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_2.setObjectName("lineEdit_2")
        regex=QtCore.QRegExp("[0-9_]+")
        validator = QtGui.QRegExpValidator(regex)
        self.lineEdit_2.setValidator(validator)
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_3.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 15px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 15px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 15px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_3.setObjectName("pushButton_3")
        axis = pg.DateAxisItem(orientation='bottom')
        self.graphicsView_2 = PlotWidget(self.tab_2,axisItems={'bottom': axis})
       #fenêtre Affichage mesures
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.comboBox_10 = QtWidgets.QComboBox(self.tab_2)
        #Choix eps' eps"
        self.comboBox_10.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_10.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_10.setObjectName("comboBox_10")
        self.comboBox_10.addItem("")
        self.comboBox_10.addItem("")
        self.checkBox_9 = QtWidgets.QCheckBox(self.tab_2)
         #Exporter
        self.checkBox_9.setObjectName("checkBox_9")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab_2)
         #suivant page 2
        self.pushButton_4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_4.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 15px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 15px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 15px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_4.setObjectName("pushButton_4")


         ######### TAB 3 #########



        self.tabWidget.addTab(self.tab_2, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.label_15 = QtWidgets.QLabel(self.tab_7)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.graphicsView_7 = PlotWidget(self.tab_7)
                 #Fenetre courbe temporelle
        self.graphicsView_7.setObjectName("graphicsView_7")
        self.comboBox_3 = QtWidgets.QComboBox(self.tab_7)
        self.comboBox_3.setObjectName("comboBox_3")
                 #selectionner echantillon
        self.comboBox_3.addItem("")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab_7)
        self.lineEdit_4.setReadOnly(True)
        self.lineEdit_4.setObjectName("lineEdit_4")
        
        self.checkBox_1 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_1.setObjectName("checkBox_1")
                 #Choix echelle
        
        self.checkBox_13 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_13.setObjectName("checkBox_13")
        
        self.pushButton_19 = QtWidgets.QPushButton(self.tab_7)
                 #Afficher
        self.pushButton_19.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_19.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 5em;\n"
"    padding: 6px;}")
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_20 = QtWidgets.QPushButton(self.tab_7)
                 #Exporter Marqueur
        self.pushButton_20.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_20.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_20.setObjectName("pushButton_20")
        
        
        
        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab_7)
        self.lineEdit_5.setObjectName("lineEdit_5")
        
        self.lineEdit_50 = QtWidgets.QLineEdit(self.tab_7)
        self.lineEdit_50.setObjectName("lineEdit_50")
                 #Choix profondeur a exporter
        self.label_17 = QtWidgets.QLabel(self.tab_7)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        
        self.label_170 = QtWidgets.QLabel(self.tab_7)
        self.label_170.setFont(font)
        self.label_170.setObjectName("label_170")
        

        ######### TAB 4 #########



        self.tabWidget.addTab(self.tab_7, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
                 #Texte mise à jour
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_3)
                 #Parcourir
        self.pushButton_6.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        font = QtGui.QFont()
        font.setPointSize(1)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(70)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet("QPushButton{\n"
"    background-color: darkblue;\n"
"    color: rgb(255, 255, 255);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(41, 128, 185);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: darkblue;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 24px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
                 #Texte nb eps
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.tab_3)
                 #fenêtre acq eps
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.label_10 = QtWidgets.QLabel(self.tab_3)
                 #Texte diam
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.tab_3)
                 #fenêtre acq diam
        self.lineEdit_8.setText("")
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.pushButton_13 = QtWidgets.QPushButton(self.tab_3)
                 #Actualiser/Ajouter
        self.pushButton_13.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        font = QtGui.QFont()
        font.setPointSize(1)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(70)
        self.pushButton_13.setFont(font)
        self.pushButton_13.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_13.setObjectName("pushButton_13")
        self.label_11 = QtWidgets.QLabel(self.tab_3)
                 #Texte affichage
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(70)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.comboBox_6 = QtWidgets.QComboBox(self.tab_3)
                 #Selectionner ref trans
        self.comboBox_6.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_6.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_7 = QtWidgets.QComboBox(self.tab_3)
                 #Selectionner mod phase
        self.comboBox_7.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_7.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")

        self.label_4 = QtWidgets.QLabel(self.tab_3)
                 #Texte mise à jour
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(70)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.pushButton_14 = QtWidgets.QPushButton(self.tab_3)
                 #Afficher
        self.pushButton_14.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        font = QtGui.QFont()
        font.setPointSize(1)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_14.setFont(font)
        self.pushButton_14.setStyleSheet("QPushButton{\n"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:hover{\n"
"    background-color: rgb(244, 208, 63);\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}"
"QPushButton:pressed{"
"    background-color: orange;\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-radius: 10px;\n"
"    border-color: rgb(234, 237, 237);\n"
"    font: bold 20px;\n"
"    min-width: 10em;\n"
"    padding: 6px;}")
        self.pushButton_14.setObjectName("pushButton_14")
        self.listWidget_2 = QtWidgets.QListWidget(self.tab_3)
                 #affichage nom fichiers 
        self.listWidget_2.setObjectName("listWidget_2")
        self.graphicsView_3 = PlotWidget(self.tab_3)
                 #Fenetre graphe simu
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.line = QtWidgets.QFrame(self.tab_3)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.tab_3)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.tab_2)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.tab_2)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.tab_2)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        
        self.lineEdit_7.setValidator(validator)
        self.lineEdit_8.setValidator(validator)


        ######### NAVBAR #########



        self.tabWidget.addTab(self.tab_3, "")
        self.verticalLayout_2.addWidget(self.tabWidget)
        self.gridLayout_5.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.resizeEvent = self.resizeEvent
        
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DIA"))
        PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
        MainWindow.setWindowIcon(QtGui.QIcon(PATH_WHERE_EXECUTED+'/resources/logo.png')) 
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_0), _translate("MainWindow", "Antenne"))
        self.label_0.setText(_translate("MainWindow", "Bienvenue sur le DIA !"))
        
        self.pushButton_7.setText(_translate("MainWindow", "Parcourir"))
        self.pushButton_7.clicked.connect(self.browseSlot)
        self.comboBox_8.setItemText(0, _translate("MainWindow", "Reflexion"))
        self.comboBox_8.setItemText(1, _translate("MainWindow", "Transmission"))
        self.comboBox_9.setItemText(0, _translate("MainWindow", "Module"))
        self.comboBox_9.setItemText(1, _translate("MainWindow", "Phase"))
        self.comboBox_9.setItemText(2, _translate("MainWindow", "Temporelle"))
        self.pushButton_2.setText(_translate("MainWindow", "Tout sélectionner"))
        self.pushButton_2.clicked.connect(self.checkAll)
        self.pushButton_8.setText(_translate("MainWindow", "Afficher"))
        self.pushButton_8.clicked.connect(self.plot)
        self.comboBox_10.setItemText(0, _translate("MainWindow", "eps\'"))
        self.comboBox_10.setItemText(1, _translate("MainWindow", "eps\'\'"))
        self.comboBox_0.setItemText(0, _translate("MainWindow", "Antenne Vivaldi Antipodale"))
        self.comboBox_0.setItemText(1, _translate("MainWindow", "Antenne Bow Tie"))

        LIST_MET=get_liste_met()
        for i in range(len(LIST_MET)):
            self.comboBox.setItemText(i, _translate("MainWindow", LIST_MET[i]))
        
        self.pushButton_0.setText(_translate("MainWindow", "Suivant"))
        self.pushButton_0.clicked.connect(self.change_tab)
        self.pushButton.setText(_translate("MainWindow", "Suivant"))
        self.pushButton.clicked.connect(self.change_tab)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Affichage Mesures"))
        self.tabWidget.setTabToolTip(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Sélection et affichage des courbes"))
        self.label.setText(_translate("MainWindow", "Diamètre"))
        self.label_2.setText(_translate("MainWindow", "Choix des métriques"))
        self.checkBox_2.setText(_translate("MainWindow", "Fréquence de la première résonance"))
        self.checkBox_10.setText(_translate("MainWindow", "Fréquence de la deuxième résonance"))
        self.checkBox_11.setText(_translate("MainWindow", "Fréquence de la troisième résonance"))
        self.checkBox_3.setText(_translate("MainWindow", "Surface"))
        self.checkBox_4.setText(_translate("MainWindow", "Amplitude Minimum 500MHz"))
        self.checkBox_5.setText(_translate("MainWindow", "Fréquence Minimum 500MHz"))
        self.checkBox_6.setText(_translate("MainWindow", "Dérivée Minimum 500MHz"))
        self.checkBox_7.setText(_translate("MainWindow", "Derivée Maximum 500MHz"))
        self.checkBox_8.setText(_translate("MainWindow", "Transmission Valeur à 0.4GHz"))
        self.checkBox_12.setText(_translate("MainWindow", "Transmission dans le sol"))
        self.label_3.setText(_translate("MainWindow", "Nombre de voisins"))
        self.pushButton_3.setText(_translate("MainWindow", "Calculer"))
        self.pushButton_3.clicked.connect(self.calcul_eps)
        self.checkBox_9.setText(_translate("MainWindow", "Exporter les données météo"))
        self.pushButton_4.setText(_translate("MainWindow", "Exporter"))
        self.pushButton_4.clicked.connect(self.exportation)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Estimation Permittivité"))
        self.label_4.setText(_translate("MainWindow", "SIMULATION : Mise à jour "))
        self.pushButton_6.setText(_translate("MainWindow", "Parcourir"))
        self.pushButton_6.clicked.connect(self.browseSlot2)
        self.label_9.setText(_translate("MainWindow", "Diamètre"))
        self.label_10.setText(_translate("MainWindow", "Nombre de eps"))
        self.pushButton_13.setText(_translate("MainWindow", "Exporter les métriques"))
        self.pushButton_13.clicked.connect(self.simuMetriques)
        self.label_15.setText(_translate("MainWindow", "Echantillon"))
        self.pushButton_19.setText(_translate("MainWindow", "Afficher"))
        self.pushButton_19.clicked.connect(self.afficherProfondeur)
    
        self.pushButton_20.setText(_translate("MainWindow", "Exporter le marqueur"))
        self.pushButton_20.clicked.connect(self.afficherMarqueur)
        self.label_17.setText(_translate("MainWindow", "Position"))
        self.label_170.setText(_translate("MainWindow", "Temps en ns"))
        self.checkBox_1.setText(_translate("MainWindow", "Echelle temporelle"))
        
        #self.pushButton_5.setText(_translate("MainWindow", "Echelle distancielle"))
        #self.pushButton_5.clicked.connect(self.distance_scale)
        self.checkBox_13.setText(_translate("MainWindow", "Echelle distancielle"))
        
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Courbes Profondeur et Marqueurs"))
        self.label_11.setText(_translate("MainWindow", "SIMULATION : Affichage"))
        self.comboBox_6.setItemText(0, _translate("MainWindow", "Reflexion"))
        self.comboBox_6.setItemText(1, _translate("MainWindow", "Transmission"))
        self.comboBox_7.setItemText(0, _translate("MainWindow", "Module"))
        self.comboBox_7.setItemText(1, _translate("MainWindow", "Phase"))
        self.pushButton_14.setText(_translate("MainWindow", "Afficher"))
        self.pushButton_14.clicked.connect(self.afficherSimu)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Données Simulation"))
        
        self.pushButton_27.setText(_translate("MainWindow", "Supprimer"))
        self.pushButton_27.clicked.connect(self.delet)
        self.pushButton_25.setText(_translate("MainWindow", "Tout désélectionner"))
        self.pushButton_25.clicked.connect(self.decoch)
        

    def resizeEvent(self, event):
        # Récupère la nouvelle taille de la fenêtre
        new_width = event.size().width()
        new_height = event.size().height()

        font = QtGui.QFont()
        font.setPointSize(int(7/700*new_width))
        font2 = QtGui.QFont()
        font2.setPointSize(int(11/700*new_width))
        font0 = QtGui.QFont()
        font0.setPointSize(int(20/700*new_width))

        # Calcule les nouvelles dimensions et positions des boutons        
        
        self.pushButton_0.setGeometry(QtCore.QRect(int(490/700*new_width),int(395/500*new_height),int(70/700*new_width),int(40/500*new_height)))
        self.pushButton_7.setGeometry(QtCore.QRect(int(20/700*new_width),int(10/500*new_height),int(30/700*new_width),int(50/500*new_height)))
        self.pushButton_27.setGeometry(QtCore.QRect(int(50/700*new_width),int(360/500*new_height),int(50/700*new_width),int(30/500*new_height)))
        self.pushButton_25.setGeometry(QtCore.QRect(int(50/700*new_width),int(400/500*new_height),int(50/700*new_width),int(30/500*new_height)))
        self.pushButton_2.setGeometry(QtCore.QRect(int(50/700*new_width),int(320/500*new_height),int(50/700*new_width),int(30/500*new_height)))
        self.pushButton_8.setGeometry(QtCore.QRect(int(400/700*new_width),int(10/500*new_height),int(50/700*new_width),int(50/500*new_height)))
        self.pushButton.setGeometry(QtCore.QRect(int(490/700*new_width),int(395/500*new_height),int(70/700*new_width),int(40/500*new_height)))
        self.pushButton_3.setGeometry(QtCore.QRect(int(60/700*new_width),int(405/500*new_height),int(100/700*new_width),int(35/500*new_height)))
        self.pushButton_4.setGeometry(QtCore.QRect(int(410/700*new_width),int(405/500*new_height),int(100/700*new_width),int(35/500*new_height)))
        self.pushButton_19.setGeometry(QtCore.QRect(int(480/700*new_width),int(20/500*new_height),int(50/700*new_width),int(35/500*new_height)))
        self.pushButton_20.setGeometry(QtCore.QRect(int(350/700*new_width),int(390/500*new_height),int(50/700*new_width),int(40/500*new_height)))
        self.pushButton_6.setGeometry(QtCore.QRect(int(10/700*new_width),int(10/500*new_height),int(200/700*new_width),int(40/500*new_height)))
        self.pushButton_13.setGeometry(QtCore.QRect(int(30/700*new_width),int(380/500*new_height),int(201/700*new_width),int(40/500*new_height)))
        self.pushButton_14.setGeometry(QtCore.QRect(int(30/700*new_width),int(260/500*new_height),int(201/700*new_width),int(40/500*new_height)))
        
        self.comboBox_8.setGeometry(QtCore.QRect(int(20/700*new_width),int(70/500*new_height),int(110/700*new_width),int(20/500*new_height)))
        self.comboBox_8.setFont(font)
        self.comboBox_9.setGeometry(QtCore.QRect(int(150/700*new_width),int(70/500*new_height),int(110/700*new_width),int(20/500*new_height)))
        self.comboBox_9.setFont(font)
        self.comboBox.setGeometry(QtCore.QRect(int(170/700*new_width),int(30/500*new_height),int(80/700*new_width),int(22/500*new_height)))
        self.comboBox.setFont(font)
        self.comboBox_10.setGeometry(QtCore.QRect(int(440/700*new_width),int(10/500*new_height),int(91/700*new_width),int(28/500*new_height)))
        self.comboBox_10.setFont(font)
        self.comboBox_3.setGeometry(QtCore.QRect(int(140/700*new_width),int(10/500*new_height),int(180/700*new_width),int(22/500*new_height)))
        self.comboBox_3.setFont(font)
        self.comboBox_6.setGeometry(QtCore.QRect(int(30/700*new_width),int(220/500*new_height),int(110/700*new_width),int(20/500*new_height)))
        self.comboBox_6.setFont(font)
        self.comboBox_7.setGeometry(QtCore.QRect(int(170/700*new_width),int(220/500*new_height),int(110/700*new_width),int(20/500*new_height)))
        self.comboBox_7.setFont(font)
        self.comboBox_0.setGeometry(QtCore.QRect(int(200/700*new_width),int(330/500*new_height),int(300/700*new_width),int(50/500*new_height)))
        self.comboBox_0.setFont(font)
        
        self.label.setGeometry(QtCore.QRect(int(40/700*new_width),int(30/500*new_height),int(100/700*new_width),int(21/500*new_height)))
        self.label.setFont(font2)
        self.label_2.setGeometry(QtCore.QRect(int(30/700*new_width),int(60/500*new_height),int(400/700*new_width),int(60/500*new_height)))
        self.label_2.setFont(font2)
        self.label_3.setGeometry(QtCore.QRect(int(30/700*new_width),int(370/500*new_height),int(400/700*new_width),int(31/500*new_height)))
        self.label_3.setFont(font2)
        self.label_15.setGeometry(QtCore.QRect(int(20/700*new_width),int(20/500*new_height),int(120/700*new_width),int(21/500*new_height)))
        self.label_15.setFont(font2)
        self.label_17.setGeometry(QtCore.QRect(int(20/700*new_width),int(380/500*new_height),int(200/700*new_width),int(30/500*new_height)))
        self.label_17.setFont(font2)
        self.label_170.setGeometry(QtCore.QRect(int(20/700*new_width),int(410/500*new_height),int(200/700*new_width),int(30/500*new_height)))
        self.label_170.setFont(font2)
        self.label_9.setGeometry(QtCore.QRect(int(10/700*new_width),int(140/500*new_height),int(100/700*new_width),int(21/500*new_height)))
        self.label_9.setFont(font)
        self.label_10.setGeometry(QtCore.QRect(int(140/700*new_width),int(140/500*new_height),int(200/700*new_width),int(20/500*new_height)))
        self.label_10.setFont(font)
        self.label_11.setGeometry(QtCore.QRect(int(10/700*new_width),int(170/500*new_height),int(300/700*new_width),int(51/500*new_height)))
        self.label_11.setFont(font2)
        self.label_4.setGeometry(QtCore.QRect(int(10/700*new_width),int(330/500*new_height),int(311/700*new_width),int(50/500*new_height)))
        self.label_4.setFont(font2)
        self.label_0.setGeometry(QtCore.QRect(int(160/700*new_width),int(30/500*new_height),int(400/700*new_width),int(50/500*new_height)))
        self.label_0.setFont(font0)
        
        self.graphicsView.setGeometry(QtCore.QRect(int(290/700*new_width),int(70/500*new_height),int(371/700*new_width),int(311/500*new_height)))
        self.graphicsView_2.setGeometry(QtCore.QRect(int(300/700*new_width),int(50/500*new_height),int(330/700*new_width),int(300/500*new_height)))
        self.graphicsView_7.setGeometry(QtCore.QRect(int(10/700*new_width),int(70/500*new_height),int(650/700*new_width),int(300/500*new_height)))
        self.graphicsView_3.setGeometry(QtCore.QRect(int(325/700*new_width),int(50/500*new_height),int(340/700*new_width),int(350/500*new_height)))
        
        self.checkBox_2.setGeometry(QtCore.QRect(int(20/700*new_width),int(110/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_2.setFont(font)
        self.checkBox_10.setGeometry(QtCore.QRect(int(20/700*new_width),int(135/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_10.setFont(font)
        self.checkBox_11.setGeometry(QtCore.QRect(int(20/700*new_width),int(160/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_11.setFont(font)
        self.checkBox_3.setGeometry(QtCore.QRect(int(20/700*new_width),int(185/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_3.setFont(font)
        self.checkBox_4.setGeometry(QtCore.QRect(int(20/700*new_width),int(210/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_4.setFont(font)
        self.checkBox_5.setGeometry(QtCore.QRect(int(20/700*new_width),int(235/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_5.setFont(font)
        self.checkBox_6.setGeometry(QtCore.QRect(int(20/700*new_width),int(260/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_6.setFont(font)
        self.checkBox_7.setGeometry(QtCore.QRect(int(20/700*new_width),int(285/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_7.setFont(font)
        self.checkBox_8.setGeometry(QtCore.QRect(int(20/700*new_width),int(310/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_8.setFont(font)
        self.checkBox_12.setGeometry(QtCore.QRect(int(20/700*new_width),int(310/500*new_height),int(500/700*new_width),int(31/500*new_height)))
        self.checkBox_12.setFont(font)
        self.checkBox_9.setGeometry(QtCore.QRect(int(390/700*new_width),int(360/500*new_height),int(300/700*new_width),int(31/500*new_height)))
        self.checkBox_9.setFont(font)
        self.checkBox_1.setGeometry(QtCore.QRect(int(330/700*new_width),int(5/500*new_height),int(300/700*new_width),int(40/500*new_height)))
        self.checkBox_1.setFont(font)
        self.checkBox_13.setGeometry(QtCore.QRect(int(330/700*new_width),int(30/500*new_height),int(300/700*new_width),int(40/500*new_height)))
        self.checkBox_13.setFont(font)
        
        self.lineEdit_2.setGeometry(QtCore.QRect(int(220/700*new_width),int(370/500*new_height),int(100/700*new_width),int(31/500*new_height)))
        self.lineEdit_2.setFont(font)
        self.lineEdit_4.setGeometry(QtCore.QRect(int(140/700*new_width),int(40/500*new_height),int(180/700*new_width),int(22/500*new_height)))
        self.lineEdit_4.setFont(font)
        self.lineEdit_5.setGeometry(QtCore.QRect(int(200/700*new_width),int(380/500*new_height),int(100/700*new_width),int(30/500*new_height)))
        self.lineEdit_5.setFont(font)
        self.lineEdit_50.setGeometry(QtCore.QRect(int(200/700*new_width),int(410/500*new_height),int(100/700*new_width),int(30/500*new_height)))
        self.lineEdit_50.setFont(font)
        self.lineEdit_7.setGeometry(QtCore.QRect(int(70/700*new_width),int(140/500*new_height),int(61/700*new_width),int(20/500*new_height)))
        self.lineEdit_7.setFont(font)
        self.lineEdit_8.setGeometry(QtCore.QRect(int(240/700*new_width),int(140/500*new_height),int(61/700*new_width),int(20/500*new_height)))
        self.lineEdit_8.setFont(font)
        
        
        self.line.setGeometry(QtCore.QRect(int(17/700*new_width),int(160/500*new_height),int(291/700*new_width),int(20/500*new_height)))
        self.line.setFont(font)
        self.line_2.setGeometry(QtCore.QRect(int(20/700*new_width),int(310/500*new_height),int(291/700*new_width),int(20/500*new_height)))
        self.line_2.setFont(font)
        self.line_3.setGeometry(QtCore.QRect(int(10/700*new_width),int(60/500*new_height),int(291/700*new_width),int(20/500*new_height)))
        self.line_3.setFont(font)
        self.line_4.setGeometry(QtCore.QRect(int(10/700*new_width),int(303/500*new_height),int(291/700*new_width),int(20/500*new_height)))
        self.line_4.setFont(font)
        self.line_5.setGeometry(QtCore.QRect(int(10/700*new_width),int(355/500*new_height),int(291/700*new_width),int(20/500*new_height)))
        self.line_5.setFont(font)
        
        self.listWidget.setGeometry(QtCore.QRect(int(10/700*new_width),int(100/500*new_height),int(271/700*new_width),int(200/500*new_height)))
        self.listWidget.setFont(font)
        self.listWidget_2.setGeometry(QtCore.QRect(int(10/700*new_width),int(60/500*new_height),int(300/700*new_width),int(70/500*new_height)))
        self.listWidget_2.setFont(font)
        
        
        PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
        image_path=PATH_WHERE_EXECUTED + '\couleur.png'
        qpixmap = QPixmap(image_path)
        image_width = int(qpixmap.width()/1.5)
        image_height = int(qpixmap.height()/1.5)
        qpixmap = qpixmap.scaled(int(image_width/700*new_width), int(image_height/500*new_height), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        #self.lbl_img.setGeometry(QtCore.QRect(int(170/700*new_width),int(100/500*new_height),int(image_width/700*new_width),int(image_height/500*new_height)))
        self.lbl_img.setGeometry(int(230/700*new_width),int(100/500*new_height),int(image_width/700*new_width),int(image_height/500*new_height))
        self.lbl_img.setPixmap(qpixmap)
        
        
         

    def browseSlot(self):
        filename = QtWidgets.QFileDialog.getOpenFileNames(filter="XLS (*.xlsx *.xls)")
        path=filename[0]
        for p in path:
            item = QListWidgetItem(p)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.listWidget.addItem(item)

    def browseSlot2(self):
        self.listWidget_2.clear()
        filename=QFileDialog.getOpenFileNames(filter="CSV(*.csv)")
        path=filename[0]
        if (len(path)==2):
            for p in path:
                item = QListWidgetItem(p)
                self.listWidget_2.addItem(item)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Selectionner 2 fichiers (S11 et S21) pour un même diamètre')
            msg.setWindowTitle("Echec !")
            msg.exec_()

    
    def handleComboBoxChange(self, index):        
        text = self.comboBox.currentText()
        if text.isnumeric():
            diametre = int(text)
        else :
            diametre = -1
        
        if (diametre==205 or diametre==206):
            self.checkBox_8.setVisible(True)
            self.checkBox_12.setVisible(False)
        elif (diametre==0):
            self.checkBox_12.setVisible(True)
            self.checkBox_8.setVisible(False)
        else:
            self.checkBox_8.setVisible(False)
            self.checkBox_12.setVisible(False)
    def simuMetriques(self):
        nbr_eps=int(self.lineEdit_8.text())+1
        diam=int(self.lineEdit_7.text())

        for i in range(2):
            p=str(self.listWidget_2.item(i).text())
            if 'S11' in p:
                df_S11=pd.read_csv(p)
            if 'S21' in p:
                df_S21=pd.read_csv(p)
        
        extractionnsaveMet(nbr_eps,diam,df_S11,df_S21)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText('Extraction des métriques de diamètre '+str(diam)+' accomplie avec succès ! \nBase de données de '+ str(nbr_eps-1) + ' eps')
        msg.setWindowTitle("Succès !")
        msg.exec_()
    
    def afficherSimu(self):
        self.graphicsView_3.clear()
        for i in range(2):
            p=str(self.listWidget_2.item(i).text())
            if 'S11' in p:
                df_S11=pd.read_csv(p)
            if 'S21' in p:
                df_S21=pd.read_csv(p)

        nb_eps=int(self.lineEdit_8.text())+1

        if(self.comboBox_6.currentText()=='Reflexion'):
            if(self.comboBox_7.currentText()=='Module'):
                for i in range(1,nb_eps):
                    self.graphicsView_3.plot(df_S11[df_S11.columns[0]],df_S11[df_S11.columns[i]])
                    self.graphicsView_3.setLabels(bottom='Frequence [GHz]', left='dB')

            if(self.comboBox_7.currentText()=='Phase'):
                for i in range(nb_eps,2*nb_eps-1):
                    self.graphicsView_3.plot(df_S11[df_S11.columns[0]],df_S11[df_S11.columns[i]])
                    self.graphicsView_3.setLabels(bottom='Frequence [GHz]', left='deg')            

        if(self.comboBox_6.currentText()=='Transmission'):
            if(self.comboBox_7.currentText()=='Module'):
                for i in range(1,nb_eps):
                    self.graphicsView_3.plot(df_S11[df_S11.columns[0]],df_S21[df_S21.columns[i]])
                    self.graphicsView_3.setLabels(bottom='Frequence [GHz]', left='dB')

            if(self.comboBox_7.currentText()=='Phase'):
                for i in range(nb_eps,2*nb_eps-1):
                    self.graphicsView_3.plot(df_S11[df_S11.columns[0]],df_S21[df_S21.columns[i]])
                    self.graphicsView_3.setLabels(bottom='Frequence [GHz]', left='deg')


    def checkAll(self):
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Unchecked:
                self.listWidget.item(index).setCheckState(QtCore.Qt.Checked)  


    def change_tab(self):
        self.tabWidget.setCurrentIndex(1)


    def calcul_eps(self):
        self.graphicsView_2.clear()
        
        diametre=int(self.comboBox.currentText())

        metriques=[]
        
        if(self.checkBox_5.isChecked()):
            metriques.append("bande freq 2")
        if(self.checkBox_3.isChecked()):
            metriques.append("r_m surface bande_freq 1")
        if(self.checkBox_4.isChecked()):
            metriques.append("r_m minimum bande_freq 1")
        if(self.checkBox_2.isChecked()):
            metriques.append("r_m freq minimum bande_freq 1")
            
        if(self.checkBox_10.isChecked()):
            metriques.append("r_m freq minimum bande_freq 2")
        if(self.checkBox_11.isChecked()):
            metriques.append("r_m freq minimum bande_freq 3")
        if(self.checkBox_6.isChecked()):
            metriques.append("r_m minimum derivee bande_freq 1")
        if(self.checkBox_7.isChecked()):
            metriques.append("r_m maximum derivee bande_freq 1")
        print("metriques",metriques)
        metriques_2=[]
        if(self.checkBox_12.isChecked()):
            metriques_2.append("t_m valeur a 3 GHz")

        if(self.checkBox_8.isChecked()):
            metriques_2.append("t_m valeur a 0.4 GHz")
   
        if (self.lineEdit_2.text() == ""):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Nombre de voisins non renseigné \nValeur par défaut = 1')
            msg.setWindowTitle("Attention")
            msg.exec_()
            nbr_voisins=1
        else:
            nbr_voisins=int(self.lineEdit_2.text())

        PATH=[]
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked:
                if ("donneesHTL" not in str(self.listWidget.item(index).text())):
                    PATH.append(str(self.listWidget.item(index).text()))
        
        if len(PATH)>0:  

            df_terrain,data_list=read_xls(PATH)
            #print("data_list2==",data_list)
            df_terrain = df_terrain[~df_terrain.index.duplicated(keep='first')]

            filter_col_r_m = [col for col in df_terrain if col.startswith('r_m')]
            filter_col_r_p = [col for col in df_terrain if col.startswith('r_p')]
            filter_col_t_m = [col for col in df_terrain if col.startswith('t_m')]
            filter_col_t_p = [col for col in df_terrain if col.startswith('t_p')]

            df_r_m=df_terrain[filter_col_r_m].T
            df_r_p=df_terrain[filter_col_r_p].T
            df_t_m=df_terrain[filter_col_t_m].T
            df_t_p=df_terrain[filter_col_t_p].T
                

            if(self.comboBox_10.currentText()=="eps'"):

                if len(metriques)==0:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText('Aucune mesure n\'a été sélectionnée pour eps\'.')
                    msg.setWindowTitle("Echec")
                    msg.exec_()
                else:                    

                    eps=predict_eps(diametre,metriques,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain,data_list)
                    if (len(eps)==1):
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Information)
                        msg.setText("eps'  =  "+str(eps[0]))
                        msg.setWindowTitle("Valeur eps'")
                        msg.exec_()

                    else:

                        import time 
                        import datetime 

                        timestamps=[]
                        for string in df_terrain.index:
                            element = datetime.datetime.strptime(string,"%Y-%m-%d %H:%M:%S") 
                            
                            tuple = element.timetuple() 
                            timestamp = time.mktime(tuple)
                            timestamps.append(timestamp) 
                        
                        self.graphicsView_2.plot(timestamps,eps)
                        self.graphicsView_2.setLabels(bottom='Date ', left="eps'")


            if(self.comboBox_10.currentText()=="eps''"):
                if len(metriques_2)==0:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText('Aucune mesure n\'a été sélectionnée pour eps\'\'.')
                    msg.setWindowTitle("Echec")
                    msg.exec_()
                else:

                    eps2=predict_eps2(diametre,metriques_2,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain)
                    if (len(eps2)==1):
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Information)
                        msg.setText("eps''  =  "+str(eps2[0]))
                        msg.setWindowTitle("Valeur eps''")
                        msg.exec_()

                    else:

                        import time 
                        import datetime 

                        timestamps=[]
                        for string in df_terrain.index:
                            element = datetime.datetime.strptime(string,"%Y-%m-%d %H:%M:%S") 
                            
                            tuple = element.timetuple() 
                            timestamp = time.mktime(tuple)
                            timestamps.append(timestamp) 
                        
                        self.graphicsView_2.plot(timestamps,eps2)
                        self.graphicsView_2.setLabels(bottom='Date ', left="eps''")

            _translate = QtCore.QCoreApplication.translate

            for i in range(len(df_terrain.index)):
                self.comboBox_3.addItem("")
                self.comboBox_3.setItemText(i, _translate("MainWindow", df_terrain.index[i]))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Aucun fichier selectionné')
            msg.setWindowTitle("Echec")
            msg.exec_()
            
    def delet(self):
        
        #print(self.listWidget.count())
        n=self.listWidget.count()

        index=0
        while index<n:
            #print(index,n)
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked :
                self.listWidget.takeItem(self.listWidget.row(self.listWidget.item(index)))
                n=n-1
            else :
                index = index+1

    def decoch(self):
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked :
                self.listWidget.item(index).setCheckState(QtCore.Qt.Unchecked)

    def afficherProfondeur(self):
        self.graphicsView_7.clear()
        Date=self.comboBox_3.currentText()

        PATH=[]
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked:
                if ("donneesHTL" not in str(self.listWidget.item(index).text())):
                    PATH.append(str(self.listWidget.item(index).text()))

        
        if len(PATH)>0:  
            df_terrain, data_list=read_xls(PATH)
            df_terrain = df_terrain[~df_terrain.index.duplicated(keep='first')]
            filter_col_r = [col for col in df_terrain if col.startswith('r temps')]
            
            df_r=df_terrain[filter_col_r].T
            courbe_temp=df_r[Date]
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Aucun fichier sélectionné!')
            msg.setWindowTitle("Echec")
            msg.exec_()

        diametre= int(self.comboBox.currentText())

        metriques=[]
        if(self.checkBox_5.isChecked()):
            metriques.append("bande freq 2")
        if(self.checkBox_3.isChecked()):
            metriques.append("r_m surface bande_freq 1")
        if(self.checkBox_4.isChecked()):
            metriques.append("r_m minimum bande_freq 1")
        if(self.checkBox_2.isChecked()):
            metriques.append("r_m freq minimum bande_freq 1")
            
        if(self.checkBox_10.isChecked()):
            metriques.append("r_m freq minimum bande_freq 2")
        if(self.checkBox_11.isChecked()):
            metriques.append("r_m freq minimum bande_freq 3")
        if(self.checkBox_6.isChecked()):
            metriques.append("r_m minimum derivee bande_freq 1")
        if(self.checkBox_7.isChecked()):
            metriques.append("r_m maximum derivee bande_freq 1")

        if (self.lineEdit_2.text() == ""):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Nombre de voisins non renseigné \nValeur par défaut = 1')
            msg.setWindowTitle("Attention")
            msg.exec_()
            nbr_voisins=1
        else:
            nbr_voisins=int(self.lineEdit_2.text())

        df_terrain, data_list=read_xls(PATH)
        df_terrain = df_terrain[~df_terrain.index.duplicated(keep='first')]

        filter_col_r_m = [col for col in df_terrain if col.startswith('r_m')]
        filter_col_r_p = [col for col in df_terrain if col.startswith('r_p')]
        filter_col_t_m = [col for col in df_terrain if col.startswith('t_m')]
        filter_col_t_p = [col for col in df_terrain if col.startswith('t_p')]

        df_r_m=df_terrain[filter_col_r_m].T
        df_r_p=df_terrain[filter_col_r_p].T
        df_t_m=df_terrain[filter_col_t_m].T
        df_t_p=df_terrain[filter_col_t_p].T

        eps=predict_eps(diametre,metriques,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain,data_list)

        eps_prof=eps[df_terrain.index.to_list().index(Date)]

        self.lineEdit_4.setText("eps'  =  "+str(eps_prof))

        vit_lum=299792458

        coeff=(vit_lum*100)/(np.sqrt(eps_prof))

        filter_col_r = [col for col in df_terrain if col.startswith('r temps')]

        temps1=[]
        for str_ in filter_col_r:
            temps1.append(str_.replace('r temps ', ''))
        temps2=[]
        for str_ in temps1:
            temps2.append(str_.replace(' s', ''))
        xt=[float(i) for i in temps2]
        
            

        profondeur = np.dot(coeff,xt)

        # self.graphicsView_7.plot(profondeur,courbe_temp)
        # self.graphicsView_7.setLabels(bottom='Profondeur (cm)', left='')
        
        if(self.checkBox_1.isChecked() and self.checkBox_13.isChecked()):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Erreur !')
            msg.setWindowTitle("")
            msg.exec_()
            
            self.checkBox_1.setCheckable(False)
            self.checkBox_13.setCheckable(False)
        
        if(self.checkBox_1.isChecked()):
            self.graphicsView_7.plot(xt,courbe_temp)
            self.graphicsView_7.setLabels(bottom='temps (ns)', left='')
            
        if(self.checkBox_13.isChecked()):
            self.graphicsView_7.plot(profondeur,courbe_temp)
            self.graphicsView_7.setLabels(bottom='Profondeur (cm)', left='')
        
        
        # return profondeur,xt
        
    # def time_scale(self)
        # #rt = afficherProfondeur(self)
        # self.graphicsView_7.plot(afficherProfondeur(self)[1],courbe_temp)
        
    # def distance_scale(self)
        # #rt = afficherProfondeur(self)
        # self.graphicsView_7.plot(afficherProfondeur(self)[0],courbe_temp)

    def afficherMarqueur(self):
        PATH=[]
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked:
                if ("donneesHTL" not in str(self.listWidget.item(index).text())):
                    PATH.append(str(self.listWidget.item(index).text()))
        
        if len(PATH)>0:  
            df_terrain,data_list=read_xls(PATH)
            df_terrain = df_terrain[~df_terrain.index.duplicated(keep='first')]

            filter_col_r = [col for col in df_terrain if col.startswith('r temps')]

            temps1=[]
            for str_ in filter_col_r:
                temps1.append(str_.replace('r temps ', ''))
            temps2=[]
            for str_ in temps1:
                temps2.append(str_.replace(' s', ''))
            xt=[float(i) for i in temps2]

            vit_lum=299792458

            eps_prof_=self.lineEdit_4.text()
            eps_prof=float(eps_prof_.replace("eps'  =  ",""))
            coeff=(vit_lum*100)/(np.sqrt(eps_prof))
            profondeur = np.dot(coeff,xt)
            coeff=(vit_lum*100)/(np.sqrt(eps_prof))
            if self.lineEdit_50.text()=="": 
                pos=float(self.lineEdit_5.text())
                #self.lineEdit_50.setText(str(pos*np.sqrt(eps_prof)/(vit_lum*100)))
                
            if self.lineEdit_5.text()=="" :
                temps=float(self.lineEdit_50.text())*pow(10,-9)
                pos=vit_lum*temps*100/(np.sqrt(eps_prof))
                #self.lineEdit_5.setText(str())
            
            idx=find_nearest(profondeur,pos)
            PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
            df_terrain["r temps "+str(xt[idx])+" s"].to_csv(PATH_WHERE_EXECUTED+"/EXPORTED/Marqueur"+str(xt[idx])+" s"+".csv",index=True)
                    
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Fichier marqueur exporté avec succès !')
            msg.setWindowTitle("Succès")
            msg.exec_()

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Aucun fichier selectionné')
            msg.setWindowTitle("Echec")
            msg.exec_()

    def exportation(self):
        
        diametre=int(self.comboBox.currentText())

        metriques=[]
        if(self.checkBox_5.isChecked()):
            metriques.append("bande freq 2")
        if(self.checkBox_3.isChecked()):
            metriques.append("r_m surface bande_freq 1")
        if(self.checkBox_4.isChecked()):
            metriques.append("r_m minimum bande_freq 1")
        if(self.checkBox_2.isChecked()):
            metriques.append("r_m freq minimum bande_freq 1")
            
        if(self.checkBox_10.isChecked()):
            metriques.append("r_m freq minimum bande_freq 2")
        if(self.checkBox_11.isChecked()):
            metriques.append("r_m freq minimum bande_freq 3")
        if(self.checkBox_6.isChecked()):
            metriques.append("r_m minimum derivee bande_freq 1")
        if(self.checkBox_7.isChecked()):
            metriques.append("r_m maximum derivee bande_freq 1")

        metriques_2=[]
        if(self.checkBox_8.isChecked()):
            metriques_2.append("t_m valeur a 0.4 GHz")


        if (self.lineEdit_2.text() == ""):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Nombre de voisins non renseigné \nValeur par défaut = 1')
            msg.setWindowTitle("Attention")
            msg.exec_()
            nbr_voisins = 1
        else:
            nbr_voisins=int(self.lineEdit_2.text())

        PATH=[]
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked:
                if ("donneesHTL" not in str(self.listWidget.item(index).text())):
                    PATH.append(str(self.listWidget.item(index).text()))
                else:
                    PATH_METEO=str(self.listWidget.item(index).text())

        if len(PATH)>0:  

            df_terrain, data_list=read_xls(PATH)
            df_terrain = df_terrain[~df_terrain.index.duplicated(keep='first')]

            filter_col_r_m = [col for col in df_terrain if col.startswith('r_m')]
            filter_col_r_p = [col for col in df_terrain if col.startswith('r_p')]
            filter_col_t_m = [col for col in df_terrain if col.startswith('t_m')]
            filter_col_t_p = [col for col in df_terrain if col.startswith('t_p')]

            df_r_m=df_terrain[filter_col_r_m].T
            df_r_p=df_terrain[filter_col_r_p].T
            df_t_m=df_terrain[filter_col_t_m].T
            df_t_p=df_terrain[filter_col_t_p].T


            eps=predict_eps(diametre,metriques,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain,data_list)
            eps2=predict_eps2(diametre,metriques_2,nbr_voisins,df_r_m,df_r_p,df_t_m,df_t_p,df_terrain)
            

                      
            
            #################################################################
            ## A ENLEVER
            
            freq=get_freq(df_r_m)
            t_m_mesure=[]
            for ech in range(0,df_terrain.shape[0]):
                t_m = adapt_form_df(df_t_m,ech,freq)
                f_mesure = t_m['Freq [GHz]'].iloc[(t_m['Freq [GHz]'] - 0.394).abs().idxmin()]
                t_m_mesure.append(t_m.loc[t_m['Freq [GHz]'] == f_mesure, ech].iloc[0])
            PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
            t_m_new = pd.DataFrame({'Colonne': t_m_mesure})
            t_m_new.to_csv(PATH_WHERE_EXECUTED+'/EXPORTED/df_t_m.csv',index=False)
            
            ##################################################################
            
            
            

            df_to_export=pd.DataFrame(data=df_terrain.index)
            df_to_export["eps'"]=eps
            df_to_export["eps''"]=eps2

            if(self.checkBox_9.isChecked()):
                df_meteo=pd.read_excel(PATH_METEO)
                df_to_export_with_meteo=pd.merge(df_to_export, df_meteo, how='left', on='Date')
                PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
                df_to_export_with_meteo.to_csv(PATH_WHERE_EXECUTED+'/EXPORTED/df_exported_diam_'+str(diametre)+'_met_'+str(len(metriques+metriques_2))+'_vois_'+str(nbr_voisins)+'.csv',index=False)
            else:
                PATH_WHERE_EXECUTED = os.path.dirname(os.path.realpath(__file__))
                df_to_export.to_csv(PATH_WHERE_EXECUTED+'/EXPORTED/df_exported_diam_'+str(diametre)+'_met_'+str(len(metriques+metriques_2))+'_vois_'+str(nbr_voisins)+'.csv',index=False)

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Exportation des données réussie !')
            msg.setWindowTitle("Succès !")
            msg.exec_()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Aucun fichier selectionné')
            msg.setWindowTitle("Echec")
            msg.exec_()


    def plot(self):
        self.graphicsView.clear()
        PATH=[]
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked:
                if ("donneesHTL" not in str(self.listWidget.item(index).text())):
                    PATH.append(str(self.listWidget.item(index).text()))
        if len(PATH)>0:  
            df_terrain,data_list=read_xls(PATH)
            
            df_terrain = df_terrain[~df_terrain.index.duplicated(keep='first')]
            
            filter_col_r_m = [col for col in df_terrain if col.startswith('r_m')]
            filter_col_r_p = [col for col in df_terrain if col.startswith('r_p')]
            filter_col_t_m = [col for col in df_terrain if col.startswith('t_m')]
            filter_col_t_p = [col for col in df_terrain if col.startswith('t_p')]
            filter_col_r = [col for col in df_terrain if col.startswith('r temps')]
            filter_col_t = [col for col in df_terrain if col.startswith('t temps')]
            
            df_r_m=df_terrain[filter_col_r_m].T
            df_r_p=df_terrain[filter_col_r_p].T
            df_t_m=df_terrain[filter_col_t_m].T
            df_t_p=df_terrain[filter_col_t_p].T
            df_r=df_terrain[filter_col_r].T
            df_t=df_terrain[filter_col_t].T
            freq1=[]
            for str_ in df_r_m.index:
                freq1.append(str_.replace('r_m frequence ', ''))
            freq2=[]
            for str_ in freq1:
                freq2.append(str_.replace(' Hz', ''))
            freq=[float(i)*1e-9 for i in freq2]

            temps1=[]
            for str_ in df_r.index:
                temps1.append(str_.replace('r temps ', ''))
            temps2=[]
            for str_ in temps1:
                temps2.append(str_.replace(' s', ''))
            temps=[float(i) for i in temps2]

            if(self.comboBox_8.currentText()=='Reflexion'):
                if(self.comboBox_9.currentText()=='Module'):
                    for i in range(df_r_m.shape[1]):
                        self.graphicsView.plot(freq,df_r_m[df_r_m.columns[i]])
                        self.graphicsView.setLabels(bottom='Frequence [GHz]', left='dB')

                if(self.comboBox_9.currentText()=='Phase'):
                    for i in range(df_r_p.shape[1]):
                        self.graphicsView.plot(freq,df_r_p[df_r_p.columns[i]])
                        self.graphicsView.setLabels(bottom='Frequence [GHz]', left='deg')

                if(self.comboBox_9.currentText()=='Temporelle'):
                    for i in range(df_r.shape[1]):
                        self.graphicsView.plot(temps,df_r[df_r.columns[i]])
                        self.graphicsView.setLabels(bottom='Temps',left='')
                

            if(self.comboBox_8.currentText()=='Transmission'):
                if(self.comboBox_9.currentText()=='Module'):
                    for i in range(df_t_m.shape[1]):
                        self.graphicsView.plot(freq,df_t_m[df_t_m.columns[i]])
                        self.graphicsView.setLabels(bottom='Frequence [GHz]', left='dB')

                if(self.comboBox_9.currentText()=='Phase'):
                    for i in range(df_t_p.shape[1]):
                        self.graphicsView.plot(freq,df_t_p[df_t_p.columns[i]])
                        self.graphicsView.setLabels(bottom='Frequence [GHz]', left='deg')

                if(self.comboBox_9.currentText()=='Temporelle'):
                    for i in range(df_t.shape[1]):
                        self.graphicsView.plot(temps,df_t[df_t.columns[i]])
                        self.graphicsView.setLabels(bottom='Temps',left='')
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Aucun fichier selectionné')
            msg.setWindowTitle("Echec")
            msg.exec_()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
