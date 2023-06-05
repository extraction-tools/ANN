import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

testmodel = tf.keras.models.load_model(r'C:\Users\Devin\Desktop\Spin Physics Work\Deuteron\trained_model_1M_v5.h5')


def Predict(X,Polarization):
    #X = np.array(X)
    acc = []
    X = np.reshape(X,(1,500))
    Y = testmodel.predict(X)
    Y = Y.reshape((len(Y),))
    g = 0.05
    s = 0.04
    bigy=(3-s)**0.5
    labelfontsize = 30

    def cosal(x,eps):
        return (1-eps*x-s)/bigxsquare(x,eps)


    def c(x):
        return ((g**2+(1-x-s)**2)**0.5)**0.5


    def bigxsquare(x,eps):
        return (g**2+(1-eps*x-s)**2)**0.5


    def mult_term(x,eps):
        return float(1)/(2*np.pi*np.sqrt(bigxsquare(x,eps)))


    def cosaltwo(x,eps):
        return ((1+cosal(x,eps))/2)**0.5


    def sinaltwo(x,eps):
        return ((1-cosal(x,eps))/2)**0.5


    def termone(x,eps):
        return np.pi/2+np.arctan((bigy**2-bigxsquare(x,eps))/((2*bigy*(bigxsquare(x,eps))**0.5)*sinaltwo(x,eps)))


    def termtwo(x,eps):
        return np.log((bigy**2+bigxsquare(x,eps)+2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps))/(bigy**2+bigxsquare(x,eps)-2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps)))

    def icurve(x,eps):
        return mult_term(x,eps)*(2*cosaltwo(x,eps)*termone(x,eps)+sinaltwo(x,eps)*termtwo(x,eps))
        
    xvals = np.linspace(-6,6,500)
    yvals = icurve(xvals,1)/10
    yvals2 = icurve(-xvals,1)/10
    x_arr = np.linspace(31.5,32.5,500)
    
    p = Polarization
    p_pred = Y
    accuracy = ((p - Y)/p)*100
    acc.append(accuracy)
    #snr = SNR[i]
    r = (np.sqrt(4-3*p**(2))+p)/(2-2*p)
    r_pred = (np.sqrt(4-3*p_pred**(2))+p_pred)/(2-2*p_pred)
    center = 250
    length = range(500)
    norm_array = []
    norm_array_pred = []
    for x in length:
        norm_array = np.append(norm_array,(x - center)*(12/500))  
        norm_array_pred = norm_array
    Iplus = icurve(norm_array,1)
    Iminus = icurve(norm_array,-1)
    ratio = Iminus/Iplus
    array = r*Iminus
    array_pred = r_pred*Iminus
    array_flipped = np.flip(array)
    array_pred_flipped = np.flip(array_pred)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = p
    element_1_pred = array_pred_flipped + Iminus
    sum_array_pred = np.sum(array_pred_flipped)*(12/500)
    element_2_pred = 1/sum_array_pred
    element_3_pred = p_pred
    result = element_1*element_2*element_3
    result_pred = element_1_pred*element_2_pred*element_3_pred
    result_new = result.reshape(500,)
    result_pred_new = result_pred.reshape(500,)
    #arr = np.array(X.iloc[[i]]).reshape(500,)
    #return result_pred_new
    return p_pred
    #return p_pred
    # plt.figure()
    #plt.plot(norm_array,arr)
    # plt.plot(norm_array,result_pred_new)
    # plt.xlim(-3,3)
    # plt.xlabel('R')
    # plt.ylabel('Intensity')
    # plt.legend(['True','Predicted'])
    # plt.ylim(0,.015)
    # p = p*100
    # plt.title("P_true:" + "%.4f %%" %(p) + "\n P_pred:" + str(np.round(p_pred*100,3)) + "%%"+ "\n SNR:" + str(np.round(snr,3)))
    # plt.savefig('Test_Plots_TE_v3/Model_Prediction__10K_V1_After_No'+ str(i) +'.png')
    # plt.figure()
    #plt.plot(norm_array,result_new)
    # arr = np.array(X.iloc[[i]]).reshape(500,)
    # plt.plot(x_arr,arr)
    #plt.ylim(0,1)
    #plt.xlim(-3,3)
    # plt.xlabel('Frequency')
    # plt.ylabel('Intensity')
    #plt.legend(['True','Sample_Noise'])
    # plt.title("P_pred:" + str(np.round(p_pred*100,3)) + "%%" + "\n P_true:" + "%.4f %%" %(p) + "\n SNR:" + str(np.round(snr,3)))
    # plt.savefig('Test_Plots_TE_v3/_Model_Prediction_10K_V1_Before_No' + str(i) + '.png')