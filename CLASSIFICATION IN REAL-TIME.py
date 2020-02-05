import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path='dataset/mini_dataset'
path2='dataset/mini_dataset_aug'
fixed_size = tuple((500, 500))

feature=[]
result=[]
labels=[]

feature2=[]
def extract(im2):
    histr=cv2.calcHist([im2],[0,1,2],None,[8,8,8],[0,255,0,255,0,255])
    return histr.flatten()

for fol in os.listdir(path): #train_color
    new_path=path+'/'+fol
    for image in os.listdir(new_path):
        des=new_path+'/'+image
        im = cv2.imread(des)
        if im is not None:
            im = cv2.resize(im, fixed_size)
            feature=extract(im)
            result.append(feature)
            labels.append(fol)
        else:
            print('Failed to open the file')

new_result=np.array(result)
new_labels=np.array(labels)

'''time_length = 30.0
fps=25
frame_seq = 249
frame_no = (frame_seq /(time_length*fps))'''
font=cv2.FONT_HERSHEY_SIMPLEX
pos=(10,50)
fontScale=1
fontColor=(255,255,255)
lineType=2
im=cv2.imread("C:/Users/asus/.spyder-py3/final-year-project/emoji.jpg")
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    feature=extract(frame)
    new_feature=np.array(feature)
    feature2=np.reshape(new_feature,(-1,512))
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(new_result,new_labels)
    y_pred=clf.predict(feature2)
    text=str(y_pred)
    cv2.putText(frame,text,pos,font,fontScale,fontColor,lineType)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()




'''font=cv2.FONT_HERSHEY_SIMPLEX
pos=(10,50)
fontScale=2
fontColor=(255,255,255)
lineType=2
for img in os.listdir(path2): #test_color
    img_loc=path2+'/'+img
    im = cv2.imread(img_loc)
    feature=extract(im)
    new_feature=np.array(feature)
    feature2=np.reshape(new_feature,(-1,512))
    classifier=DecisionTreeClassifier()  
    classifier.fit(new_result,new_labels)
    y_pred=classifier.predict(feature2)
    #print("\nClass:",y_pred)
    text=str(y_pred)
    cv2.putText(im,text,pos,font,fontScale,fontColor,lineType)
    plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    plt.show()'''
    

'''X_train,X_test,y_train,y_test=train_test_split(new_result,new_labels,test_size=0.3,random_state=9)
classifier=DecisionTreeClassifier()  
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test) 
print('Feature extraction (color)')
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
#print("\nClass:",y_pred)'''
