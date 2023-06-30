from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QListWidget,QListWidgetItem,QMessageBox
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os, os.path
import re
import time
import datetime

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib import pyplot

import sys


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2000, 1200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 30, 400, 41))
        self.label.setObjectName("label")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(30,80, 500, 400))
        self.listWidget.setObjectName("listWidget")

        self.checkBox_mois = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_mois.setGeometry(QtCore.QRect(1200, 750, 150, 31))
        self.checkBox_mois.setObjectName("checkBox_1")

        self.checkBox_sem = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_sem.setGeometry(QtCore.QRect(1400,750, 150, 31))
        self.checkBox_sem.setObjectName("checkBox_sem")

        self.checkBox_jour = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_jour.setGeometry(QtCore.QRect(1600, 750, 150, 31))
        self.checkBox_jour.setObjectName("checkBox_jour")

        self.checkBox_moy = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_moy.setGeometry(QtCore.QRect(1100,850, 170, 31))
        self.checkBox_moy.setObjectName("checkBox_moy")

        self.checkBox_max = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_max.setGeometry(QtCore.QRect(1300, 850, 170, 31))
        self.checkBox_max.setObjectName("checkBox_max")

        self.checkBox_min = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_min.setGeometry(QtCore.QRect(1500, 850, 150, 31))
        self.checkBox_min.setObjectName("checkBox_min")

        self.checkBox_ecart = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_ecart.setGeometry(QtCore.QRect(1700, 850, 170, 31))
        self.checkBox_ecart.setObjectName("checkBox_ecart")
        
        self.checkBox_glob = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_glob.setGeometry(QtCore.QRect(1400, 1050, 170, 31))
        self.checkBox_glob.setObjectName("checkBox_glob")




        self.listWidget_2 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_2.setGeometry(QtCore.QRect(1100,50, 800, 600))
        self.listWidget_2.setObjectName("listWidget_2")


        self.listWidget_2 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_2.setGeometry(QtCore.QRect(550,80, 500, 400))
        self.listWidget_2.setObjectName("listWidget_2")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 820, 300, 50))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_del = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_del.setGeometry(QtCore.QRect(500, 700, 300, 50))
        self.pushButton_del.setObjectName("pushButton_del")

        self.pushButton_raff = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_raff.setGeometry(QtCore.QRect(100, 700, 300, 50))
        self.pushButton_raff.setObjectName("pushButton_raff")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 820, 300, 50))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(140,1020, 300, 50))
        self.pushButton_6.setObjectName("pushButton_6")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(500, 1020, 300, 50))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(650, 820, 300, 50))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(650, 30, 400, 41))
        self.label_2.setObjectName("label_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 785, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "          Fichiers à traiter "))
        self.pushButton.setText(_translate("MainWindow", "selectionner tout"))
        ui.pushButton.clicked.connect(self.checkAll)

        self.checkBox_glob.setText(_translate("MainWindow", "allure globale"))

        self.checkBox_mois.setText(_translate("MainWindow", "mois"))
        self.checkBox_sem.setText(_translate("MainWindow", "semaine"))
        self.checkBox_jour.setText(_translate("MainWindow", "journée"))

        self.checkBox_max.setText(_translate("MainWindow", "maximum"))
        self.checkBox_min.setText(_translate("MainWindow", "minimum"))
        self.checkBox_moy.setText(_translate("MainWindow", "moyenne"))
        self.checkBox_ecart.setText(_translate("MainWindow", "écartype"))




        self.pushButton_raff.setText(_translate("MainWindow", "Rafraichir"))
        self.pushButton_del.setText(_translate("MainWindow", "Supprimer le fichier"))
        ui.pushButton_raff.clicked.connect(self.browseSlot_2)
        ui.pushButton_del.clicked.connect(self.delet)

        self.pushButton_2.setText(_translate("MainWindow", "Désélectionner tout"))
        ui.pushButton_2.clicked.connect(self.decoch)
        self.pushButton_3.setText(_translate("MainWindow", "Fusionner les fichiers "))

        self.pushButton_6.setText(_translate("MainWindow", "Parcourir"))
        ui.pushButton_6.clicked.connect(self.browseSlot)


        ui.pushButton_3.clicked.connect(self.fusionner)
        self.pushButton_4.setText(_translate("MainWindow", "Générer la courbe "))
        ui.pushButton_4.clicked.connect(self.plot_globale)
        self.label_2.setText(_translate("MainWindow", "Traçage des courbes "))

    def browseSlot(self):

        filename=QFileDialog.getOpenFileNames(filter="csv(*.csv)")
        path=filename[0]
        for p in path:
            item = QListWidgetItem(p)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.listWidget.addItem(item)


    def browseSlot_2(self):
        q=0
        self.listWidget_2.clear()
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked:
                q=q+1
                df=pd.read_csv(self.listWidget.item(index).text())
                columns = df.columns
                QListWidgetItem("fichier"+str(index+1), self.listWidget_2)
                for i in range(len(columns)):
                    if columns[i]!='Unnamed: 0':
                        item = QListWidgetItem(columns[i])
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        item.setCheckState(QtCore.Qt.Unchecked)
                        self.listWidget_2.addItem(item)




    def fusionner(self):
        datafr1 = pd.read_csv(self.listWidget.item(0).text(),index_col='Date',parse_dates=True)
        datafr2 = pd.read_csv(self.listWidget.item(0).text(),index_col='Date',parse_dates=True)
        for index in range(self.listWidget.count()):
            PATH=str(self.listWidget.item(index).text())
            datafr1 = pd.read_csv(PATH)
            datafr2 = pd.concat([datafr1, datafr2]).sort_values(by='Date')
        datafr3=datafr2.drop_duplicates().reset_index()
        q=len(datafr3.columns)
        p=0
        while p < q:
            if datafr3.columns[p] == 'Unnamed: 0' or datafr3.columns[p]=='index' :
                del datafr3[datafr3.columns[p]]
                q=q-1
            else :
                p=p+1
        datafr3.dropna(inplace=True)

        PATH=r"C:\Users\fdemonto-admin\Documents\labo\françois\geophysique\cc7.csv"


        datafr3.to_csv(PATH,index=False)

        item = QListWidgetItem(PATH)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Unchecked)
        self.listWidget.addItem(item)


    def checkAll(self):
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Unchecked:
                self.listWidget.item(index).setCheckState(QtCore.Qt.Checked)

    def delet(self):
        n=self.listWidget.count()
        index=0
        while index<n:

            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked :
                self.listWidget.takeItem(self.listWidget.row(self.listWidget.item(index)))
                n=n-1
            else:
                index = index+1

    def decoch(self):
        for index in range(self.listWidget.count()):
            if self.listWidget.item(index).checkState() == QtCore.Qt.Checked :
                self.listWidget.item(index).setCheckState(QtCore.Qt.Unchecked)

    def nettoye(path):    #pour nettoyer le fichier
        df = pd.read_csv(path,index_col='Date',parse_dates=True)
        data=df.reset_index()
        q=data.shape[1]
        p=0
        while p < q:

            if data.columns[p] == 'Unnamed: 0' or data.columns[p]=='index' :
                del data[data.columns[p]]
                q=q-1
            else :
                p=p+1
        return df

    def recup_path(self):
        liste_path=[]
        for index in range (self.listWidget.count()) :
            if self.listWidget.item(index).checkState():
                liste_path.append(str(self.listWidget.item(index).text()))  #liste contenant les chemin des fichiers selectionnés
        return liste_path

    def recup_base(self):   #moyenne , ou min max ...
        if self.checkBox_jour.isChecked():
            l='D'
        if self.checkBox_mois.isChecked():
            l='M'
        if self.checkBox_sem.isChecked():
            l='W'
        return l

    def plot_globale(self):
        l=[]
        tw=0
        liste_path=self.recup_path()
        n=len(liste_path)
        p=0
        if self.checkBox_glob.isChecked() :
            n=0
            fig, ax1 = plt.subplots()

            for i in range (len(liste_path)) :  #parcours des fichiers
                df = pd.read_csv(liste_path[i],index_col='Date',parse_dates=True)
                data=df.reset_index()
                q=data.shape[1]
                p=0
                while p < q:

                    if data.columns[p] == 'Unnamed: 0' or data.columns[p]=='index' :
                        del data[data.columns[p]]
                        q=q-1
                    else :
                        p=p+1

                for index in range(n+1,data.shape[1]+n+1):  #parcours des colonnes du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        tw=tw+1
                        dates=[]
                        import datetime
                    

                        for string in data['Date']:
                            element = datetime.datetime.strptime(str(string),"%Y-%m-%d %H:%M:%S")
                            dates.append(element)
                            
                        if tw==2 :
                            #ax2=pyplot.gca().twinx()
                           # pl2=ax2.scatter(dates,data[data.columns[index-n-1]],s=2,c='r',label=data.columns[index-n-1])

                            #ax2.set_ylabel(data.columns[index-n-1],fontsize=28)
                            #ax2.legend(loc="upper right",fontsize=18)
                          ax2 = ax1.twinx()
                          ax2.set_ylabel(data.columns[index-n-1],fontsize=28,color='red')
                          ax2.scatter(dates,data[data.columns[index-n-1]],color='red')
                          ax2.legend(loc="upper right",fontsize=18)
                          ax2.tick_params(axis ='y', labelcolor = 'red',labelsize=28)
  
   
                                              

                        else:
                         # plt_1=figure(figsize=(22,6))
                         ax1.set_xlabel('Date',fontsize=28)
                         ax1.set_ylabel(data.columns[index-n-1],fontsize=28,color='blue')
                         ax1.scatter(dates,data[data.columns[index-n-1]],color='blue')
                         ax1.tick_params(axis ='y', labelcolor = 'blue',labelsize=28)
                         ax1.tick_params(axis ='x', labelcolor = 'black',labelsize=28)

                                

                           # plt.scatter(dates,data[data.columns[index-n-1]],s=2,c='b',label=data.columns[index-n-1])
                            
                           # plt.gca().update(dict(xlabel='Date', ylabel=data.columns[index-n-1]))
                           # plt.legend(loc="upper left",fontsize=18)
                           
                           #plt.show()    

                            
#1=
                n=n+data.shape[1]+1
        if self.checkBox_moy.isChecked() :
            if n==2:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                data = pd.read_csv(liste_path[1],index_col='Date',parse_dates=True)
                q=data.shape[1]
                q2=df.shape[1]
                p=0
                p2=0
                y=[]
                while p < q:

                    if data.columns[p] == 'Unnamed: 0' or data.columns[p]=='index' :
                        del data[data.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                while p2 < q2:

                    if df.columns[p2] == 'Unnamed: 0' or df.columns[p2]=='index' :
                        del df[df.columns[p2]]
                        q2=q2-1
                    else :
                        p2=p2+1

                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                plt.figure(figsize=(36,36))
                plt.subplot(211)
                df[y[0]].resample(self.recup_base()).mean().plot(label="moyenne de "+y[0],fontsize=28,lw=6,ls=':',alpha=0.8)
                plt.legend(fontsize=18)


                plt.subplot(212)
                data[y[1]].resample(self.recup_base()).mean().plot(label="moyenne de "+y[1],fontsize=28,lw=6,ls='--',alpha=0.8)
                plt.legend(fontsize=18)
            if n==1:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                q=df.shape[1]
                p=0
                y=[]
                while p < q:

                    if df.columns[p] == 'Unnamed: 0' or df.columns[p]=='index' :
                        del df[df.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                coche=0
                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                        coche=coche+1

                plt.figure(figsize=(36,36))
                for index in range (coche):
                    df[y[index]].resample(self.recup_base()).mean().plot(label="moyenne de "+y[index],lw=6,ls=':',alpha=0.8)
                    plt.legend()
        if self.checkBox_max.isChecked() :
            if n==2:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                data = pd.read_csv(liste_path[1],index_col='Date',parse_dates=True)
                q=data.shape[1]
                q2=df.shape[1]
                p=0
                p2=0
                y=[]
                while p < q:

                    if data.columns[p] == 'Unnamed: 0' or data.columns[p]=='index' :
                        del data[data.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                while p2 < q2:

                    if df.columns[p2] == 'Unnamed: 0' or df.columns[p2]=='index' :
                        del df[df.columns[p2]]
                        q2=q2-1
                    else :
                        p2=p2+1

                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                plt.figure(figsize=(36,36))
                plt.subplot(211)
                df[y[0]].resample(self.recup_base()).max().plot(label="maximum de "+y[0],fontsize=28,lw=6,ls=':',alpha=0.8)
                plt.legend(fontsize=18)


                plt.subplot(212)
                data[y[1]].resample(self.recup_base()).max().plot(label="maximum de "+y[1],fontsize=28,lw=6,ls='--',alpha=0.8)
                plt.legend(fontsize=18)
            if n==1:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                q=df.shape[1]
                p=0
                y=[]
                while p < q:

                    if df.columns[p] == 'Unnamed: 0' or df.columns[p]=='index' :
                        del df[df.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                coche=0
                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                        coche=coche+1

                plt.figure(figsize=(36,36))
                for index in range (coche):
                    df[y[index]].resample(self.recup_base()).max().plot(label="maximum de "+y[index],lw=6,ls=':',alpha=0.8)
                    plt.legend()
        if self.checkBox_min.isChecked() :
            if n==2:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                data = pd.read_csv(liste_path[1],index_col='Date',parse_dates=True)
                q=data.shape[1]
                q2=df.shape[1]
                p=0
                p2=0
                y=[]
                while p < q:

                    if data.columns[p] == 'Unnamed: 0' or data.columns[p]=='index' :
                        del data[data.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                while p2 < q2:

                    if df.columns[p2] == 'Unnamed: 0' or df.columns[p2]=='index' :
                        del df[df.columns[p2]]
                        q2=q2-1
                    else :
                        p2=p2+1

                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                plt.figure(figsize=(36,36))
                plt.subplot(211)
                df[y[0]].resample(self.recup_base()).min().plot(label="minimum de "+y[0],fontsize=28,lw=6,ls=':',alpha=0.8)
                plt.legend(fontsize=18)


                plt.subplot(212)
                data[y[1]].resample(self.recup_base()).min().plot(label="minimum de "+y[1],fontsize=28,lw=6,ls='--',alpha=0.8)
                plt.legend(fontsize=18)
            if n==1:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                q=df.shape[1]
                p=0
                y=[]
                while p < q:

                    if df.columns[p] == 'Unnamed: 0' or df.columns[p]=='index' :
                        del df[df.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                coche=0
                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                        coche=coche+1

                plt.figure(figsize=(36,36))
                for index in range (coche):
                    df[y[index]].resample(self.recup_base()).min().plot(label="minimum de "+y[index],lw=6,ls=':',alpha=0.8)
                    plt.legend()
        if self.checkBox_ecart.isChecked() :
            if n==2:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                data = pd.read_csv(liste_path[1],index_col='Date',parse_dates=True)
                q=data.shape[1]
                q2=df.shape[1]
                p=0
                p2=0
                y=[]
                while p < q:

                    if data.columns[p] == 'Unnamed: 0' or data.columns[p]=='index' :
                        del data[data.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                while p2 < q2:

                    if df.columns[p2] == 'Unnamed: 0' or df.columns[p2]=='index' :
                        del df[df.columns[p2]]
                        q2=q2-1
                    else :
                        p2=p2+1

                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                plt.figure(figsize=(36,36))
                plt.subplot(211)

                df[y[0]].resample(self.recup_base()).std().plot(label="écartype de "+y[0],fontsize=28,lw=6,ls=':',alpha=0.8)
                plt.legend(fontsize=18)


                plt.subplot(212)
                data[y[1]].resample(self.recup_base()).std().plot(label="écartype de "+y[1],fontsize=28,lw=6,ls='--',alpha=0.8)
                plt.legend(fontsize=18)
            if n==1:
                df = pd.read_csv(liste_path[0],index_col='Date',parse_dates=True)
                q=df.shape[1]
                p=0
                y=[]
                while p < q:

                    if df.columns[p] == 'Unnamed: 0' or df.columns[p]=='index' :
                        del df[df.columns[p]]
                        q=q-1
                    else :
                        p=p+1
                coche=0
                for index in range (self.listWidget_2.count()):  #parcours des colones du fichiers i
                    if self.listWidget_2.item(index).checkState():
                        y.append(str(self.listWidget_2.item(index).text()))
                        coche=coche+1

                plt.figure(figsize=(36,36))
                for index in range (coche):
                    df[y[index]].resample(self.recup_base()).std().plot(label="écartype de "+y[index],lw=6,ls=':',alpha=0.8)
                    plt.legend()


        plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
