#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:38:19 2018

@author: adhesh
"""
from tkinter import *
import pandas as pd
from tkinter import messagebox
window = Tk()

window.title("UI")
window.geometry('1000x1000')
lbl1=Label(window, text = "Predicting Aging Inventory",font=("Arial Bold",40))
lbl1.place(x=200,y=10)

btn1=Button(window,text="Click")
btn1.place(x=100,y=150)
btn2=Button(window,text="Click")
btn2.place(x=100,y=250)
btn3=Button(window,text="Click")
btn3.place(x=100,y=350)

lbl2=Label(window,text="ORDERS: ",font=("Arial Bold",20))
lbl2.place(x=300,y=150)

txt=Entry(window,width=30)
txt.place(x=500,y=150)

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
    
def clicked():
   order=txt.get()
   p=0
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
   label.place(x=800,y=130)
   label=Label(window,text=int(mat3[p].values))  
   label.place(x=800,y=150) 
   messagebox.showinfo("Order Places","Order Places from Factory")
btn4=Button(window,text="Click",command=clicked)
btn4.place(x=700,y=150)

window.mainloop()



   