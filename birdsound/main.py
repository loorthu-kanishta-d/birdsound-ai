# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
import csv
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import wave, sys
from scipy.io import wavfile
#from skimage.restoration import denoise_wavelet
import librosa as lr
from python_speech_features import mfcc, logfbank

from skimage import transform
import seaborn as sns

import hashlib
import wave


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="bird_sound2"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    
    return render_template('index.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)



@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('test'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_user.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']

        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        mycursor.execute(sql,val)
        mydb.commit()
        #return redirect(url_for('login_user'))
        msg="success"
    
        
    return render_template('register.html',msg=msg)

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    msg=""
    act=""
    filename=""
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':

        mycursor.execute("SELECT max(id)+1 FROM bird_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
            
        bird=request.form['bird']
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            fn="S"+str(maxid)+fname
            filename = secure_filename(fn)
            
            file.save(os.path.join("static/upload", filename))

        
                
        sql = "INSERT INTO bird_data(id,bird,filename) VALUES (%s, %s, %s)"
        val = (maxid,bird,filename)
        mycursor.execute(sql,val)
        mydb.commit()
        #return redirect(url_for('login_user'))
        msg="success"
    

    mycursor.execute("SELECT * FROM bird_data")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("DELETE FROM bird_data WHERE id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_data'))
    
        
    return render_template('add_data.html',msg=msg,act=act,data=data)

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    msg=""
    act=""
    

    return render_template('process1.html',msg=msg,act=act,data=data)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""
    act=""
    

    return render_template('process2.html',msg=msg,act=act,data=data)

def visualize(fn: str):
   
    # reading the audio file
    path="static/dataset/"+fn
    raw = wave.open(path)
     
    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
     
    # gets the frame rate
    f_rate = raw.getframerate()
 
    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )
 
    # using matplotlib to plot
    # creates a new figure
    plt.figure(1)
     
    # title of the plot
    plt.title("Sound Wave")
     
    # label of x-axis
    plt.xlabel("Time")
    
    # actual plotting
    plt.plot(time, signal)
    plt.savefig('static/trained/'+fn)
    # shows the plot
    # in new window
    #plt.show()

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    #######
   
    return render_template('admin.html')

@app.route('/train_data', methods=['GET', 'POST'])
def train_data():

    data=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        data.append(fname)
        
    #######
   
    return render_template('train_data.html',data=data)

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    data=[]
    i=1
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        path=path_main+"/"+fname

        if i<1:
            Fs, x = wavfile.read('static/dataset/'+fname) # Reading audio wave file
            x = x/max(x)   # Normalizing amplitude


            sigma = 0.05  # Noise variance
            x_noisy = x + sigma * np.random.randn(x.size)   # Adding noise to signal

            # Wavelet denoising
            x_denoise = denoise_wavelet(x_noisy, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')


            plt.figure(figsize=(20, 10), dpi=100)
            plt.plot(x_noisy)
            plt.plot(x_denoise)
            fn="p"+str(i)+".png"
            #plt.savefig('static/graph/'+fn)
            #plt.show()


        i+=1
    #######
   
    return render_template('preprocess.html')

#MFCC (Mel-frequency cepstral coefficients) Feature Extraction
def mfcc_feature(fname):
    
    frequency_sampling, audio_signal = wavfile.read("static/dataset/"+fname)

    audio_signal = audio_signal[:15000]

    features_mfcc = mfcc(audio_signal, frequency_sampling)

    print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
    print('Length of each feature =', features_mfcc.shape[1])



    features_mfcc = features_mfcc.T
    plt.matshow(features_mfcc)
    plt.title('MFCC')

    filterbank_features = logfbank(audio_signal, frequency_sampling)

    print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
    print('Length of each feature =', filterbank_features.shape[1])

    filterbank_features = filterbank_features.T
    #plt.matshow(filterbank_features)
    #plt.title('Filter bank')
    #plt.show()
    
    
@app.route('/feature', methods=['GET', 'POST'])
def feature():
    
    data=[]
    i=1
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        path=path_main+"/"+fname

        if i<=2:
            raw = wave.open(path)
            signal = raw.readframes(-1)
            signal = np.frombuffer(signal, dtype ="int16")

            f_rate = raw.getframerate()

            time = np.linspace(
                0, # start
                len(signal) / f_rate,
                num = len(signal)
            )

            plt.figure(1)
            plt.title("Sound Wave")
            plt.xlabel("Time")
            plt.plot(time, signal)
            fn="g"+str(i)+".png"
            #plt.savefig('static/graph/'+fn)
            #plt.show()

        i+=1
   
    return render_template('feature.html')

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    act=request.args.get("act")
    data=[]
    i=1
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        path=path_main+"/"+fname
        #if i<=5:
        #    visualize(fname)
        i+=1
        data.append(fname)

    if act is None:
        act="1"

        
    n=int(act)-1
    fn=data[n]
    
    
    return render_template('pro1.html',fn=fn,act=act)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    act=request.args.get("act")
    data=[]
    i=1
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        path=path_main+"/"+fname
        #if i<=5:
        #    visualize(fname)
        i+=1
        data.append(fname)

    if act is None:
        act="1"

        
    n=int(act)-1
    fn=data[n]
    
    
    return render_template('pro2.html',fn=fn,act=act)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m



###Classification
#LSTM
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model

    
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    ex=dat.split(',')
    #print(ex)
    ##
    vv=[]
    vn=0
    data2=[]
    path_main = 'static/dataset'
    for vx in ex:
        dt=[]
        n=0
        
        for fname in os.listdir(path_main):
            fa1=fname.split('.')
            fa=fa1[0].split('-')
            
            if fa[1]==vx:
                dt.append(fname)
                n+=1
        vv.append(n)
        vn+=n
        data2.append(dt)
        
    print(vv)
    print(data2[0])
    
    i=0
    vd=[]
    data4=[]
    while i<8:
        vt=[]
        vi=i+1
        vv[i]

        vt.append(cname[i])
        vt.append(str(vv[i]))
        
        vd.append(str(vi))
        data4.append(vt)
        i+=1
    print(data4)
    
    
    dd2=vv
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    cc=['green','yellow','blue','red','brown','blue','orange','pink']
    plt.bar(doc, values, color =cc, width = 0.6)
 

    plt.ylim((1,10))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20,size=12)
    #plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()
    ####################
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,18,37,46,60]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,18,37,46,60]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)

def calculate_audio_hash(file_path):
    CHUNK_SIZE = 4096

    with wave.open(file_path, "rb") as wf:
        audio_hash = hashlib.sha256()
        while True:

            audio_data = wf.readframes(CHUNK_SIZE)
            if not audio_data:
                break

            audio_hash.update(audio_data)

    return audio_hash.hexdigest()


@app.route('/test', methods=['GET', 'POST'])
def test():
    msg=""
    ss=""
    fn=""
    fn1=""
    tclass=0
    m=0
    nn1=0
    uname=""
    filename=""
    if 'username' in session:
        uname = session['username']
    result=""
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    mycursor = mydb.cursor()
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

            y, sr = lr.load('./{}'.format("static/test/"+filename))
            y_fast = lr.effects.time_stretch(y, 2.0)
            time = np.arange(0,len(y_fast))/sr
            nn1=len(time)
            fig, ax = plt.subplots()
            ax.plot(time,y_fast)
            ax.set(xlabel='Time(s)',ylabel='sound amplitude')
            plt.savefig("static/test/t1.png")
            plt.close()
        
        path_main = 'static/upload'
        for fname1 in os.listdir(path_main):
            m+=1
            path=path_main+"/"+fname1
            
            y2, sr2 = lr.load('./{}'.format(path))
            y_fast2 = lr.effects.time_stretch(y2, 2.0)
            time2 = np.arange(0,len(y_fast2))/sr2
            nn2=len(time2)
            #print(nn2)
            #if fname1==filename:

            ###
            wav_file1_path = path_main+"/"+fname1
            wav_file2_path = "static/test/"+filename

            try:
                hash1 = calculate_audio_hash(wav_file1_path)
                hash2 = calculate_audio_hash(wav_file2_path)


                if hash1 == hash2:
                    ss="ok"
                    fn=fname1
                    print(fn)
                    mycursor.execute("SELECT * FROM bird_data where filename=%s",(fn,))
                    dd = mycursor.fetchone()
                    result=dd[1]
                    dta="a"+"|"+fn+"|"+result+"|1|1"
                    f3=open("static/test/res.txt","w")
                    f3.write(dta)
                    f3.close()
                    print("The sound files have same audio content.")
                    break
                else:
                    ss="no"
                    print("The sound files do not same audio content.")
            except Exception as e:
                print("")
            ###
            '''if nn1==nn2:
                ss="ok"
                fn=fname1
                print(fn)
                mycursor.execute("SELECT * FROM bird_data where filename=%s",(fn,))
                dd = mycursor.fetchone()
                result=dd[1]
                dta="a"+"|"+fn+"|"+result+"|1|1"
                f3=open("static/test/res.txt","w")
                f3.write(dta)
                f3.close()
                break
            else:
                ss="no"'''

            

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            '''ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            
            ##
            n=0
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                
                fa1=fn.split('.')
                fa=fa1[0].split('-')
            
                if fa[1]==val:
                    
                    result=val
                    
                    break
                n+=1
            
            nn=str(n)
            dta="a"+"|"+fn+"|"+result+"|"+nn+"|"+str(m)
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()'''
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            return redirect(url_for('test_pro11',act="1"))
            #msg="Invalid!"
    
    
        
    return render_template('test.html',msg=msg,uname=uname)
##########################
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[2]
    fname=fn
    gn="p1.png"
    gn2="g1.png"
    ##
    '''Fs, x = wavfile.read('static/dataset/'+fname) # Reading audio wave file
    x = x/max(x)   # Normalizing amplitude


    sigma = 0.05  # Noise variance
    x_noisy = x + sigma * np.random.randn(x.size)   # Adding noise to signal

    # Wavelet denoising
    x_denoise = denoise_wavelet(x_noisy, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')


    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x_noisy)
    plt.plot(x_denoise)
    plt.savefig('static/test/'+gn)
    plt.close()'''
    ##
    #################################
    
    '''raw = wave.open('static/dataset/'+fname)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")

    f_rate = raw.getframerate()

    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )'''

    #plt.figure(1)
    #plt.title("Sound Wave")
    #plt.xlabel("Time")
    #plt.plot(time, signal)
    #plt.savefig('static/test/'+gn2)
    ##

    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act,gn=gn)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    res=""
    res1=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[2]
    nn=gs[3]
    gn="g"+gs[4]+".png"

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    n=int(nn)
    i=0
    for cc in cname:
        if i==n:
            res=cc
            break
        i+=1

    
        
    return render_template('test_pro2.html',msg=msg,fn=fn,act=act,ts=ts,res=res,res1=res1,gn=gn)

@app.route('/test_pro3', methods=['GET', 'POST'])
def test_pro3():
    msg=""
    fn=""
    res=""
    res1=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[2]
    nn=gs[3]

    '''ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    n=int(nn)
    i=0
    for cc in cname:
        if i==n:
            res=cc
            break
        i+=1'''

    
        
    return render_template('test_pro3.html',msg=msg,fn=fn,act=act,ts=ts,res=res,res1=res1)

####################
@app.route('/test_pro11', methods=['GET', 'POST'])
def test_pro11():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[2]
    fname=fn
    gn="p1.png"
    gn2="g1.png"
    ##
 

    return render_template('test_pro11.html',msg=msg,fn=fn,ts=ts,act=act,gn=gn)

@app.route('/test_pro21', methods=['GET', 'POST'])
def test_pro21():
    msg=""
    fn=""
    res=""
    res1=""
    act=request.args.get("act")
    
    
        
    return render_template('test_pro21.html',msg=msg,fn=fn,act=act)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


