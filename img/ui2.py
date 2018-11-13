from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import time
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn import linear_model, tree, ensemble
from imblearn.pipeline import Pipeline
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import itertools

window = Tk()

window.title("UI")
window.geometry('1000x1000')
lbl1=Label(window, text = "Predicting Aging Inventory",font=("Arial Bold",40))
lbl1.place(x=200,y=10)
lbl2=Label(window,text="ORDERS: ",font=("Arial Bold",20))
lbl2.place(x=300,y=150)

txt=Entry(window,width=30)
txt.place(x=500,y=150)

format = 'jpeg'
def process(df):
	    from sklearn.preprocessing import Imputer
	    df['lead_time'] = Imputer(strategy='median').fit_transform(
	                                    df['lead_time'].values.reshape(-1, 1))
	    df = df.dropna()
	    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
	        df[col] = Imputer(missing_values=-99).fit_transform(
	                                    df[col].values.reshape(-1, 1))
	   
	    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
	               'stop_auto_buy', 'rev_stop', 'xcs_pro']:
	        df[col] = (df[col] == 'Yes').astype(int)
	    
	    from sklearn.preprocessing import normalize
	    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
	                   'forecast_6_month', 'forecast_9_month', 'min_bank',
	                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
	                   'sales_3_month', 'sales_6_month', 'sales_9_month',]
	    df[qty_related] = normalize(df[qty_related], axis=1)
	    return df
    
def roc_plot(estimators, models,dataset,y_test,X_test,ls):
    """ Plot ROC curves for each estimator.
    """
    f, ax = plt.subplots()
    for est, mdl in zip(estimators, models):
        _roc_plot(y_test,mdl.predict_proba(X_test),label=est,ax=ax,l=next(ls))
    ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
        label='Random Classifier')    
    ax.legend(loc="lower right")    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(False)
    sns.despine()
    plt.savefig('img/'+dataset+'/auc_score.'+format,format=format,bbox_inches='tight',dpi=450)
    plt.show()

def _roc_plot(y_true, y_proba, label=' ', l='-', lw=1.0, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

def pr_plot(estimators, models,dataset,y_test,X_test,ls):
    """ Plot Precision-Recall curves for each estimator.
    """
    f, ax = plt.subplots()
    for est, mdl in zip(estimators, models):
        _pr_aux(y_test,mdl.predict_proba(X_test),label=est,ax=ax,l=next(ls))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    sns.despine()
    plt.savefig('img/'+dataset+'/precision_recall.'+format,format=format,bbox_inches='tight',dpi=450)
    plt.show()  
    
def _pr_aux(y_true, y_proba, label=' ', l='-', lw=1.0, ax=None):
    precision, recall, _ = precision_recall_curve(y_true,
                                                  y_proba[:,1])
    average_precision = average_precision_score(y_true, y_proba[:,1],
                                                     average="micro")
    ax.plot(recall, precision, label='%s (area=%.3f)'%(label,average_precision),
            linestyle=l, linewidth=lw)
    
def clicked1():
	cols=range(0,23)
	train = pd.read_csv('data/kaggle/Kaggle_Training_Dataset_v2.csv', usecols=cols)
	test = pd.read_csv('data/kaggle/Kaggle_Test_Dataset_v2.csv', usecols=cols)
	df = process(train.append(test))

	sample = df.sample(5000,random_state=5)
	X_sample = sample.drop('xcs_pro',axis=1).values
	y_sample = sample['xcs_pro'].values

	df.round(6).to_csv('data/kaggle.csv',index=False)
    
def clicked2():
	n_runs=30 
	scorer = make_scorer(average_precision_score, needs_threshold=True, average="micro",)

	min_samples_leaf=5
	n_estimators=10
	criterion='entropy'
	max_depth=np.arange(3,45,5)
	max_depth=[3,4,5,7,10,15,20,30,50]
	dataset='kaggle' 
	n_folds=5
	save_run=10

	df = pd.read_csv('data/'+dataset+'.csv')
	X = df.drop(['xcs_pro','sku'],axis=1).values
	y = df['xcs_pro'].values

	print("dataset:",dataset)

	estimators = [
	    ("Logistic Regression", 'lgst', 
	    linear_model.LogisticRegression(), 
	    {'C':np.logspace(0,3,4),
	     'penalty':['l1','l2'],
	    }),   
	        
	    ("RandomForest", "rf",
	     ensemble.RandomForestClassifier(n_estimators=n_estimators,
	            min_samples_leaf=min_samples_leaf, criterion=criterion),
	    {'max_depth':max_depth,
	    }),

	    ("GradientBoosting", "gb",
	     ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
	            min_samples_leaf=min_samples_leaf),
	    {'max_depth':[10,],
	    }),
	]

	for es_name, estimator_name, est, parameters in estimators:
	    print ('\n%s\n' % ( es_name))
	    print ('Run\tEst\tScore\tAUROC\tAUPRC\tTime\tBest parameters')
	    matriz = []
	    t0 = time.time()
	    for run in range(n_runs):
	        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
	                                    test_size=0.15, random_state=run)
	        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(run*9))
	        gs = GridSearchCV(est, parameters, cv=kf,# n_iter=n_iter_search,
	                          scoring=scorer, verbose=0,n_jobs=-1)  
	        t1 = time.time()
	        gs.fit(X_train, y_train)
	 
	        y_prob0 = gs.best_estimator_.predict_proba(X_train)[:,1]
	        y_prob = gs.best_estimator_.predict_proba(X_test)[:,1]
	        
	        roc =  roc_auc_score(y_test, y_prob)
	        pr = average_precision_score(y_test, y_prob)   
	        
	        run_time = time.time() - t1
	        avg_time = run_time/gs.n_splits_
	        
	        print ("%i\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t%s" % (run, estimator_name, 
	            gs.best_score_, roc, pr, avg_time, gs.best_params_))
	        imp = []
	        mdl = gs.best_estimator_

	        if estimator_name == 'lgst':
	            imp = mdl.coef_.ravel()
	        else:
	            imp = mdl.feature_importances_
	        
	        matriz.append(
	        {   'run'           : run,
	            'estimator'     : estimator_name,         
	            'roc'           : roc,
	            'pr'            : pr,
	            'best_params'   : gs.best_params_, 
	            'avg_time'      : avg_time,
	            'importance'    : imp,
	        })
	        
	        if run == save_run:
	            path = 'results/pkl/'+dataset+'/'+estimator_name.lower() + '.pkl'        
	            joblib.dump(gs.best_estimator_, path) 
	 
	    print("Elapsed time: %0.3fs" % (time.time()-t0))
	    data = pd.DataFrame(matriz)
    
def clicked3():	
	sns.set_style('white')
	ls = itertools.cycle(['-','--','-.',':'])

	n_run=10 
	dataset = 'kaggle'
	format = 'jpeg'

	estimators = ['lgst','rf','gb']

	df = pd.read_csv('data/'+dataset+'.csv')
	X = df.drop(['xcs_pro','sku'],axis=1).values
	y = df['xcs_pro'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
	                                    test_size=0.15, random_state=n_run)

	models = []
	for est in estimators:
	    models.append(joblib.load('results/pkl/'+dataset+'/'+est+'.pkl'))

	print ('\n%s\n' % ('Area under ROC curve'))
	roc_plot(estimators, models,dataset,y_test,X_test,ls)

	print ('\n%s\n' % ('Precision-recall curve'))
	pr_plot(estimators, models,dataset,y_test,X_test,ls)
    
rs=pd.read_csv('recomendor.csv',usecols=['xcs_pro','fact_no','product_left','aging'])
matrix=[]
mat=[]
mat2=[]
mat3=[]
final=[]
j=1

for i in rs['xcs_pro'].values:
   if(i==1):
       matrix.append(j)
   j+=1  
for i in matrix:
    mat.append(rs.iloc[[i-1],[2]])
    mat2.append(rs.iloc[[i-1],[3]])
    mat3.append(rs.iloc[[i-1],[1]])
p=0   
def clicked():
   order=txt.get()
   
   q=mat[0].values
   for i in range(0,7):
       if(mat[i].values>=int(order)):
           final.append(i)
       else:
           for j in range(0,7):
              if(mat[j].values>q):
                  q=mat[j].values
                  p=j
   l=mat2[0].values       
   for j in final:
       if(mat2[j].values>l):
           l=mat2[j].values
           p=j       
   label=Label(window,text='Order Places from Factory Id:')  
   label.place(x=600,y=330)
   label=Label(window,text=int(mat3[p].values))  
   label.place(x=600,y=350) 
   messagebox.showinfo("Order Places","Order Places from Factory")
   

def clicked4():
   window.destroy()
   
   
btn1=Button(window,text="Click",command=clicked1)
btn1.place(x=100,y=150)
btn2=Button(window,text="Click",command=clicked2)
btn2.place(x=100,y=250)
btn3=Button(window,text="Click",command=clicked3)
btn3.place(x=100,y=350)
btn4=Button(window,text="Click",command=clicked)
btn4.place(x=700,y=150)
btn5=Button(window,text="Click",command=clicked4)
btn5.place(x=100,y=750)


window.mainloop()