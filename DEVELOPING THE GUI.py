from tkinter import *
from PIL import Image, ImageTk
import tkinter.ttk as ttk
import tkinter as tk
from tkinter import filedialog
import os
from shutil import copy2
import cv2
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
import mahotas as mt
import matplotlib.pyplot as plt

LARGE_FONT=("Verdana",8)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                   
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Zii")
        self.master.iconbitmap("C:/Users/asus/.spyder-py3/final-year-project/Zii.ico")
        self.pack(fill=BOTH, expand=1)
        
        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)

        edit = Menu(menu)
        edit.add_command(label="About Zii", command=self.about_dino)
        menu.add_cascade(label="Help", menu=edit)

        label=Label(text="Choose file for train dataset",font=LARGE_FONT)
        label.place(x=15,y=10)
        global folder_path
        folder_path = StringVar()
        b_button = ttk.Button(text="Browse", command=self.browse_button)
        b_button.place(x=15,y=31)
        
    def client_exit(self):
        exit()
    
    def about_dino(self):
        lbl1 = Label(self,text="About the app")
        lbl1.place(x=15,y=57)
        
    
    def browse_button(self):
            filename = filedialog.askdirectory()
            folder_path.set(filename)
            print(filename)
            path=filename
            os.mkdir('C:/data')
            path2='C:/data'
            for fol in os.listdir(path):
                os.mkdir(path2+'/'+fol)
                for image in os.listdir(path+'/'+fol):
                    source=os.path.join(path,fol,image)
                    des=os.path.join(path2,fol)
                    copy2(source,des)
                    cur_image=cv2.imread(os.path.join(des,image))
                    new_filename,ext=os.path.splitext(image)
                    npath=new_filename+'1'+ext #filename for flip
                    final_path=des+'/'+npath
                    flipim=np.fliplr(cur_image) #for flip
                    cv2.imwrite(final_path,flipim)
                    
                    npath=new_filename+'2'+ext #filename for blur
                    final_path=des+'/'+npath
                    blurImg=cv2.blur(cur_image,(10,10))
                    cv2.imwrite(final_path,blurImg)
                    
                    npath=new_filename+'4'+ext #filename for rotate
                    final_path=des+'/'+npath
                    r1=(randint(-25,25))
                    h,w=cur_image.shape[:2]
                    scale=1.0
                    center=tuple(np.array([h,w])/2)
                    M=cv2.getRotationMatrix2D(center,r1,scale)
                    rotated=cv2.warpAffine(cur_image,M,(h,w))
                    cv2.imwrite(final_path,rotated)
                    
                    npath=new_filename+'5'+ext #filename for exposure
                    final_path=des+'/'+npath
                    a=np.double(cur_image)
                    r2=randint(-50,50)
                    b=a+r2
                    exposure=np.uint8(b)
                    cv2.imwrite(final_path,exposure)
                    
                    npath=new_filename+'3'+ext #filename for noise
                    final_path=des+'/'+npath
                    row,col,ch=cur_image.shape
                    s_vs_p=0.9
                    amount=.04
                    out=cur_image
                    num_salt=np.ceil(amount*cur_image.size*s_vs_p)
                    coords=[np.random.randint(0,i-1,int(num_salt)) for i in cur_image.shape]
                    out[coords]=1
                    num_pepper=np.ceil(amount*cur_image.size*(1-s_vs_p))
                    coords=[np.random.randint(0,i-1,int(num_pepper)) for i in cur_image.shape]
                    out[coords]=1
                    cv2.imwrite(final_path,out)
            
            label=Label(text="The aug dataset is created",font=LARGE_FONT)
            label.place(x=15,y=76)
            print('The aug dataset is created')
            label=Label(text="Click the 'Start button' to start the training and predict the accuracy",font=LARGE_FONT)
            label.place(x=15,y=92)
            s_button = ttk.Button(text="Start", command=self.pred)
            s_button.place(x=15,y=114)
            #self.pred()
            
    def pred(self):
            path='C:/data'
            global_features=[]
            labels=[]
            fixed_size = tuple((500, 500))
            
            def extract_color(im2):
                histr=cv2.calcHist([im2],[0,1,2],None,[8,8,8],[0,255,0,255,0,255])
                return histr.flatten()
            
            def extract_texture(im2):
                texture=mt.features.haralick(im2)
                ht_mean=texture.mean(axis=0)
                return ht_mean.flatten()
            
            def extract_shape(im2):
                shape=cv2.HuMoments(cv2.moments(im2))
                return shape.flatten()
            i=0
            for fol in os.listdir(path): #feature extraction
                new_path=path+'/'+fol
                i=i+1
                cur_dir=fol
                for image in os.listdir(new_path):
                    des=new_path+'/'+image
                    im = cv2.imread(des)
                    img = cv2.imread(des)
                    if im is not None:
                        im = cv2.resize(im, fixed_size)
                        feature1=extract_color(im) #color
                        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        feature2=extract_texture(gray) #texture
                        feature3=extract_shape(gray) #shape
                        global_feature = np.hstack([feature1, feature2, feature3])
                        labels.append(cur_dir)
                        global_features.append(global_feature)
                    else:
                        print('Failed to open the file')
                print ("[STATUS] processed folder: {}.{}".format(i,cur_dir))
            
            global_feature=np.reshape(global_feature,(-1,532))
            Tr_global_feature=np.transpose(global_features)
            Tr2_global_feature=np.transpose(Tr_global_feature)

            self.new_rescaled_features=np.array(Tr2_global_feature)
            self.new_labels=np.array(labels)
            
            
            X_train,X_test,y_train,y_test=train_test_split(self.new_rescaled_features,self.new_labels,test_size=0.3,random_state=9)
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            print('Train_aug dataset combining all three')
            print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
            acc=metrics.accuracy_score(y_test,y_pred)
            print(acc)
            label=Label(text='Accuracy:',font=LARGE_FONT)
            label.place(x=15,y=141)
            label=Label(text=acc,font=LARGE_FONT)
            label.place(x=15,y=168)
            '''label=Label(text="Choose file for test dataset",font=LARGE_FONT)
            label.place(x=15,y=182)
            b_button = ttk.Button(text="Browse", command=self.browse_button2)
            b_button.place(x=15,y=214)'''
            
    def browse_button2(self):
        def extract_color(im2):
                histr=cv2.calcHist([im2],[0,1,2],None,[8,8,8],[0,255,0,255,0,255])
                return histr.flatten()
            
        def extract_texture(im2):
            texture=mt.features.haralick(im2)
            ht_mean=texture.mean(axis=0)
            return ht_mean.flatten()
            
        def extract_shape(im2):
            shape=cv2.HuMoments(cv2.moments(im2))
            return shape.flatten()
            
        font=cv2.FONT_HERSHEY_SIMPLEX
        pos=(10,50)
        fontScale=2
        fontColor=(255,255,255)
        lineType=2
        filename = filedialog.askdirectory()
        folder_path.set(filename)
        print(filename)
        path2=filename
        i=0
        y_val=235
        print('----Testing [STATUS]----')
        for fol in os.listdir(path2):
            new_path=path2+'/'+fol
            i=i+1
            cur_dir=fol
            for image in os.listdir(new_path):
                des=new_path+'/'+image
                im = cv2.imread(des)
                img = cv2.imread(des)
                if im is not None:
                    feature4=extract_color(im)
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    feature5=extract_texture(gray)
                    feature6=extract_shape(gray)
                    global_feature = np.hstack([feature4, feature5, feature6])
                    new_feature=np.array(global_feature)
                    main_feature=np.reshape(new_feature,(-1,532))
                    classifier=RandomForestClassifier(n_estimators=100)  
                    classifier.fit(self.new_rescaled_features,self.new_labels)
                    y_pred=classifier.predict(main_feature)
                    #print("\nClass:",y_pred)
                    text=str(y_pred)
                    cv2.putText(im,text,pos,font,fontScale,fontColor,lineType)
                    plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                    plt.show()
                    #diplaying img in tkinter [[error part]]
                    '''scale_percent = 20 # percent of original size
                    width = int(im.shape[1] * scale_percent / 100)
                    height = int(im.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                    #named_img=cv2.putText(resized,text,pos,font,fontScale,fontColor,lineType)
                    b,g,r = cv2.split(resized)
                    img = cv2.merge((r,g,b))
                    im = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=im)
                    label=Label(self,image=imgtk).pack(side=TOP, anchor=E, expand=YES)
                    pack(fill=BOTH, expand=YES)'''
                    # [[upto here]]
                else:
                    print('Failed to open the file')
            print ("[STATUS] processed folder: {}.{}".format(i,cur_dir))

root = Tk()
root.geometry("500x400")
app = Window(root)
root.mainloop()  