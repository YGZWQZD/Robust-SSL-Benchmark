import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 5
import numpy as np
# datasets=[
#         "wine",
#         "eye_movements",
#         "phoneme"
#         # "pol",
#         # "kdd_ipums_la_97-small",
#         # "bank-marketing",
#         # "MagicTelescope",
#         # "house_16H",
#         # "credit",
#         # "california",
#         # "electricity",
#         # "jannis",
#         # "MiniBooNE"
#     ]
datasets=['CIFAR10','CIFAR100']

methods = {
        # "FT_Transformer" : {
        #     "symbol": "-",
        #     "color": "dimgray",
        #     "reveal": "FT_Transformer"
        # },
        "Supervised" : {
            "symbol": "-",
            "color": "dimgray",
            "reveal": "Supervised"
        },
        "PseudoLabel": {
            "symbol": "8-",
            "color": "#33A02C",
            "reveal": "PseudoLabel"
        },
        # "ImprovedGAN" : {
        #     "symbol": ".-",
        #     "color": "goldenrod",
        #     "reveal": "ImprovedGAN"
        # },
        "PiModel":{
            "symbol": "v-",
            "color": "#D62728",
            "reveal": "PiModel"
        },
        "ICT": {
            "symbol": "h-",
            "color": "#F4C20D",
            "reveal": "ICT"
        },
        # "TemporalEnsembling":{
        #     "symbol": "^-",
        #     "color": "teal",
        #     "reveal": "TemporalEnsembling"
        # },
        "UDA":{
            "symbol": "D-",
            "color": "#911EB4",
            "reveal": "UDA"
        },
        # "MixMatch": {
        #     "symbol": "H-",
        #     "color": "purple",
        #     "reveal": "MixMatch"
        # },
        "FixMatch": {
            "symbol": "*-",
            "color": "#46F0F0",
            "reveal": "FixMatch"
        },
        "FlexMatch": {
            "symbol": "s-",
            "color": "orange",
            "reveal": "FlexMatch"
        },
        "SoftMatch": {
            "symbol": "s-",
            "color": "pink",
            "reveal": "SoftMatch"
        }
    }

labels=100
rate_list=[str(0),str(0.2),str(0.4),str(0.6),str(0.8),str(1)]
rate_list_f=[0,0.2,0.4,0.6,0.8,1]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("CIFAR10 and CIFAR100 with Inconsistent Feature Spaces", fontsize=13)
# plt.title("Office-31"+"with "+100+" labels", fontdict={"size": 26})
# plt.xlabel("Class Inconsistency", fontdict={"size": 26})
def AUC(Acc_T):
    return np.mean(np.array(Acc_T))
def Acc_T0(Acc_T):
    return Acc_T[0]
def WA(Acc_T):
    return np.min(np.array(Acc_T))

def EVM(Acc_T):
    sum_m=0
    for i in range(len(Acc_T)-1):
        sum_m+=Acc_T[i]-Acc_T[i+1] if Acc_T[i]>Acc_T[i+1] else Acc_T[i+1]-Acc_T[i]
    return sum_m/(len(Acc_T)-1)
def VS(Acc_T):
    return np.std(np.array([Acc_T[i+1]-Acc_T[i] for i in range(len(Acc_T)-1)]))
def RCC(Acc_T):
    return pearsonr(Acc_T,[i*1/(len(Acc_T)-1) for i in range(len(Acc_T))])[0]

pos=0
for dataset in datasets:
    f = open('./Feature/'+dataset+'_feature.csv', "r",encoding="utf-8")
    g=open('./Feature/'+dataset+'_stat.csv', "w",encoding="utf-8")
    r = csv.DictWriter(g,['dataset','algorithm','AUC','Acc_T0','WA','EVM','VS','RCC'])
    f_read=csv.reader(f)
    dict_mean={}
    dict_std={}
    _d={}
# for row in f_read:
#     # print(row)
#     if row[0]=='Supervised' and row[3]!='0':
#         print(row)
#         dict_mean[row[0]+str(row[1])+str(row[2])+str(row[3])]=dict_mean[row[0]+str(row[1])+str(row[2])+'0']
#         dict_std[row[0]+str(row[1])+str(row[2])+str(row[3])]=dict_std[row[0]+str(row[1])+str(row[2])+'0']
#     elif row[3]=='0':
#         _d=domain[0] if row[1]!=domain[0] else domain[1]
#         dict_mean[row[0]+str(row[1])+str(row[2])+str(row[3])]=dict_mean[row[0]+str(row[1])+_d+'0']
#         dict_std[row[0]+str(row[1])+str(row[2])+str(row[3])]=dict_std[row[0]+str(row[1])+_d+'0']
#     else:
#         dict_mean[row[0]+str(row[1])+str(row[2])+str(row[3])]=float(row[4])
#         dict_std[row[0]+str(row[1])+str(row[2])+str(row[3])] = float(row[5])    
    for row in f_read:
        print(row)
        if row[0]=='Supervised' and row[1]!='0':
            dict_mean[row[0]+str(row[1])]=dict_mean[row[0]+'0']
            dict_std[row[0]+str(row[1])]=dict_std[row[0]+'0']
        # elif row[1]=='0':
        #     if row[0]+'0' not in dict_mean.keys():
        #         dict_mean[row[0]+str(row[1])+_d[row[1]]+str(row[3])]=float(row[4])
        #         dict_std[row[0]+str(row[1])+_d[row[1]]+str(row[3])] = float(row[5])
        #     dict_mean[row[0]+str(row[1])+str(row[2])+str(row[3])]=dict_mean[row[0]+str(row[1])+_d[row[1]]+'0']
        #     dict_std[row[0]+str(row[1])+str(row[2])+str(row[3])]=dict_std[row[0]+str(row[1])+_d[row[1]]+'0']   
        #     # print(_d[row[1]])

        else:
            dict_mean[row[0]+str(row[1])]=float(row[2])
            dict_std[row[0]+str(row[1])] = float(row[3]) 

    i=pos//2
    j=pos%2
    max_mean=0
    min_mean=100
    for method in methods:
        color = methods[method]["color"]
        symbol = methods[method]["symbol"]
        reveal = methods[method]["reveal"]
        mean_list=[]
        std_list = []
        d={}
        for rate in rate_list:
            # if rate==0:
            #     mean_list.append(dict_mean[method +source+target+ '0'])
            #     std_list.append(dict_std[method + source+target+'0'])
            # elif rate==1:
            #     mean_list.append(dict_mean[method +source+target+ '1'])
            #     std_list.append(dict_std[method + source+target+'1'])
            # else:
            max_mean=max(max_mean,dict_mean[method+str(rate)])
            min_mean=min(min_mean,dict_mean[method+str(rate)])
            mean_list.append(dict_mean[method+str(rate)])
            std_list.append(dict_std[method+str(rate)])
        y=np.array(mean_list)
        # print(y)
        y=y*100
        y_std=np.array(std_list)
        # print(y_std)
        y_std=y_std*100/2
        X=rate_list_f
        axes[j].plot(X, y, symbol, label=reveal, color=color)
        d['dataset']=dataset
        d['algorithm']=method
        d['AUC']="{:.3f}".format(AUC(mean_list))
        d['Acc_T0']="{:.3f}".format(Acc_T0(mean_list))
        d['WA']="{:.3f}".format(WA(mean_list))
        d['EVM']="{:.3f}".format(EVM(mean_list))
        d['VS']="{:.3f}".format(VS(mean_list))
        d['RCC']="{:.3f}".format(RCC(mean_list))
        r.writerow(d)
        # plt.fill_between(X, y - y_std, y + y_std, color=color, alpha=0.2)

    # print(i)
    # print(j)
    axes[j].legend(loc="lower left")
    axes[j].set_xlabel("Feature Space Inconsistency t", fontdict={"size": 8})
    axes[j].set_ylabel("$Acc_T(t)(\%)$", fontdict={"size": 8})
    axes[j].grid()
    axes[j].set_xticks(rate_list_f)
    axes[j].set_yticks([min_mean*100+(max_mean-min_mean)*i*10 for i in range(-5,11,1)])
    axes[j].set_title(dataset,fontdict={"size": 8})
    # axes[i,j].savefig('iris' + '_deep' + '_class' + '_labels_' + str(labels)+'.png' , dpi=300)
    # axes[i,j].show()
    pos+=1
plt.tight_layout()
plt.savefig('Feature_CIFAR.eps',dpi=300)
plt.savefig('Feature_CIFAR.png',dpi=300)
plt.show()


