from flask import Flask, render_template, url_for, flash, redirect, request
import numpy as np
import pandas as pd
import json

df=pd.read_csv('head_count.csv')
time_arr=list(df['Time'])
worker=list(df['Head_count'])
for w in range(len(worker)):
  worker[w]=int(worker[w])

minutes=[]
seconds=[]
hour=[]

for i in time_arr:
    seconds.append(i[2:4])
    minutes.append(i[0:1])
   

length=len(seconds)
app = Flask(__name__)

# obj=variable()

@app.route('/',methods=['GET', 'POST'])
def home():
    
    return render_template('slider.html',minutes=minutes,seconds=seconds,length=length,worker=worker,hour=hour)
    



if __name__ == '__main__':
    app.run(debug=True)    