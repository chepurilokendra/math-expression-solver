#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import PIL.Image
import shutil
import sympy
from sympy import sympify


final_result=[]
# In[2]:


def read_image(img_path):
    img = cv2.imread(img_path)
    print(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img=cv2.resize(img,(1080,720))
    img = cv2.copyMakeBorder(img,20,20,150,80,cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.imwrite("system.jpg",img)
    img = cv2.imread("system.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_s=img.copy()
    img_d=img.copy()
    fresh_img=img.copy()
    # plt.imshow(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh_img = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh_img,cmap='gray')
    kernel = np.ones((3,250),np.uint8)
    dilated = cv2.dilate(thresh_img,kernel,iterations=1)
    # plt.imshow(dilated,cmap="gray")
    (contours,hierarchy)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contour_lines=sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])
    return line_counter(sorted_contour_lines,fresh_img)


# In[3]:


def line_counter(sorted_contour_lines,fresh_img):
    if len(sorted_contour_lines) > 1:
        return multi_line_segmentor(sorted_contour_lines,fresh_img)
    else:
        return single_line_segmentor(sorted_contour_lines,fresh_img)


# In[4]:


def multi_line_segmentor(sorted_contour_lines,fresh_img):
    img_s=fresh_img.copy()
    img2=fresh_img.copy()
    
    if os.path.exists("C:\\Users\sathw\Documents\Image segmentation\equation"):
        shutil.rmtree("C:\\Users\sathw\Documents\Image segmentation\equation")
    os.mkdir("C:\\Users\sathw\Documents\Image segmentation\equation")
    i=0
    for ctr in sorted_contour_lines:
        i=i+1
        print(i)
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(40,100,250),2)
        print(x,y,w,h)
        roi = img_s[y:y+h,x:x+w]
        cv2.imwrite('C:\\Users\sathw\Documents\Image segmentation\input_'+str(i)+'.jpg',roi)
        img_path = "C:\\Users\sathw\Documents\Image segmentation\input_"+str(i)+".jpg"
#     if os.path.exists("C:\\Users\sathw\Documents\Image segmentation\equation"):
#         shutil.rmtree("C:\\Users\sathw\Documents\Image segmentation\equation")
#     os.mkdir("C:\\Users\sathw\Documents\Image segmentation\equation")
        segmentor(img_path)
    return send_for_recognition()
    
    


# In[5]:


def single_line_segmentor(sorted_contour_lines,fresh_img):
    if os.path.exists("C:\\Users\sathw\Documents\Image segmentation\equation"):
        shutil.rmtree("C:\\Users\sathw\Documents\Image segmentation\equation")
    os.mkdir("C:\\Users\sathw\Documents\Image segmentation\equation")
    cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\input_1"+".jpg",fresh_img)
    img_path = "C:\\Users\sathw\Documents\Image segmentation\input_1"+".jpg"
    segmentor(img_path)
    return send_for_recognition()
    


# In[6]:


def segmentor(img_path):
    i=0
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    inp2=img.copy()
    inp_s=img.copy()
    fresh_img=img.copy()
    co_ordinates=[]
    img_gray = cv2.cvtColor(inp2,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh,cmap='gray')
    thresh_img= thresh
    kernel = np.ones((3,10),np.uint8)
    dilated = cv2.dilate(thresh_img,kernel,iterations=1)
    # plt.imshow(dilated,cmap="gray")
    (contours,hierarchy)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contour_lines=sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[0])
    
    for ctr,c in zip(sorted_contour_lines,range(len(sorted_contour_lines))):
    
        x_curr,y_curr,w_curr,h_curr = cv2.boundingRect(ctr)
        print(i,"  ", x_curr,y_curr,w_curr,h_curr)
        flag=0
        if c==0:
            x_new=x_curr
            y_new=y_curr
            w_new=w_curr
            h_new=h_curr
        else:
            if abs(x_curr-x_prev) <30:
                print("if")
                x_new = min(x_curr,x_prev)
                w_new= min(w_curr,w_prev)
                h_new= h_curr+abs(y_curr-y_prev)+h_prev
                y_new = min(y_curr,y_prev)
                print("x_curr ",x_curr," x_prev ",x_prev)
                print("y_curr ",y_curr," y_prev ",y_prev)
                print("h_curr ",h_curr," h_prev ",h_prev)
                print("x_curr ",w_curr," x_prev ",w_prev)
                co_ordinates.pop()
            else:
                x_new,y_new,w_new,h_new = x_curr,y_curr,w_curr,h_curr
                print("x_curr ",x_curr," x_prev ",x_prev)
                print("y_curr ",y_curr," y_prev ",y_prev)
                print("h_curr ",h_curr," h_prev ",h_prev)
                print("x_curr ",w_curr," x_prev ",w_prev)
        print("//////////////////////////////////////////")
        print("i ",i)
        co_ordinates.append([x_new,y_new,w_new,h_new])
        cv2.rectangle(inp2,(x_new,y_new),(x_new+w_new,y_new+h_new),(40,100,250),2)
        x_prev,y_prev,w_prev,h_prev = x_curr,y_curr,w_curr,h_curr
        
    path_eq=img_path.split("_")[1]
    p=path_eq.split(".")[0]
    generator_main(co_ordinates,p,img_path,fresh_img)


# In[7]:


def generator_main(co_ordinates,p,img_path,fresh_img):
    os.mkdir("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+str(p))
    square=cv2.imread("C:\\Users\sathw\Documents\Image segmentation\\times.jpg")
    print(square.shape)
    square = cv2.cvtColor(square,cv2.COLOR_BGR2RGB)
    k=1
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    inp2=img.copy()
    inp_s=img.copy()
    img_s=img.copy()
    for co,i in zip(co_ordinates,range(0,len(co_ordinates))):
    #     k=k+1
        flag=0
    #     k=1
        print("i =",i)
        x,y,w,h = co_ordinates[i]
        if i+1 in range(len(co_ordinates)):
            x_sq,y_sq,w_sq,h_sq=co_ordinates[i+1]
        print(x,y,w,h)

        if ((i+1) in range(len(co_ordinates))) and abs((x+w) - (x_sq)) < 30 and abs((y_sq+h_sq) - y) < 30:
            print("square")
            flag=1
            cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+str(p)+'\\seg_'+str(k+1)+".jpg",square)
            cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+str(p)+'\\seg_'+str(k+2)+".jpg",square)
            roi=img_s[co_ordinates[i+1][1]:co_ordinates[i+1][1]+co_ordinates[i+1][3],
                      co_ordinates[i+1][0]:co_ordinates[i+1][0]+co_ordinates[i+1][2]]
            cv2.rectangle(inp_s,(x_sq,y_sq),(x_sq+w_sq,y_sq+h_sq),(40,100,250),2)
            cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+str(p)+'\\seg_'+str(k+3)+".jpg",roi)
        else:
            roi=img_s[co_ordinates[i][1]:co_ordinates[i][1]+co_ordinates[i][3],
                      co_ordinates[i][0]:co_ordinates[i][0]+co_ordinates[i][2]]
            cv2.rectangle(inp_s,(x,y),(x+w,y+h),(40,100,250),2)
            cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+str(p)+'\\seg_'+str(k)+".jpg",roi)
            k=k+1
            continue



        if flag==1:    
            roi=img_s[co_ordinates[i][1]:co_ordinates[i][1]+co_ordinates[i][3],
                      co_ordinates[i][0]:co_ordinates[i][0]+co_ordinates[i][2]]
#             plt.imshow(roi)
            cv2.rectangle(inp_s,(x,y),(x+w,y+h),(40,100,250),2)
            cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+str(p)+'\\seg_'+str(k)+".jpg",roi)
            k=k+3
            flag=0
            continue
#         roi=img_s[co_ordinates[i+1][1]:co_ordinates[i+1][1]+co_ordinates[i+1][3],
#                   co_ordinates[i+1][0]:co_ordinates[i+1][0]+co_ordinates[i+1][2]]
#         cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation\equation_"+p+'\\seg_'+str(k)+".jpg",roi)
#         k=k+1


# In[8]:


def send_for_recognition():
    equations=[]
    model=tf.keras.models.load_model("C:\\Users\sathw\Documents\Image segmentation\model_epoch_5")
    
    class_names = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=',
               'A', 'C', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'ascii_124', 'b', 
               'beta', 'cos', 'd', 'delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 
               'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log',
               'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm','prime', 'q', 'rightarrow', 'sigma', 'sin', 'sqrt',
               'sum', 'tan', 'theta', 'times', 'u', 'v', 'w', 'y', 'z', '{', '}']

    for ip in os.listdir("C:\\Users\sathw\Documents\Image segmentation\equation"):
        print(ip)
        exp=""
        for eq in range(1,len(os.listdir("C:\\Users\sathw\Documents\Image segmentation\equation"+"\\"+ip))+1):
            img = cv2.imread("C:\\Users\sathw\Documents\Image segmentation\equation"+"\\"+ip+"\\"+"seg_"+str(eq)+".jpg")
            print("C:\\Users\sathw\Documents\Image segmentation\equation"+"\\"+ip+"\\"+"seg_"+str(eq)+".jpg")
            
            if img.shape[0]<15:
                img = cv2.copyMakeBorder(img,50,50,20,20,cv2.BORDER_CONSTANT,value=(255,255,255))
                cv2.imwrite("C:\\Users\sathw\Documents\Image segmentation\equation"+"\\"+ip+"\\"+"seg_"+str(eq)+".jpg",img)
                img = cv2.imread("C:\\Users\sathw\Documents\Image segmentation\equation"+"\\"+ip+"\\"+"seg_"+str(eq)+".jpg")
                
#             img = cv2.imread("C:\\Users\sathw\Documents\Image segmentation\equation"+"\\"+ip+"\\"+"seg_"+str(eq)+".jpg")
            img=cv2.resize(img,(45,45))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,70,255,0)
            img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    #         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


            print(img.shape)
    #       plt.imshow(img)
    #         plt.imshow(img)
            exp = exp + recognize(img,exp,model,class_names)
        equations.append(exp)
    print(equations)
    return conv_exp(equations)
    # return equations


# In[9]:


class_names = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=',
               'A', 'C', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'ascii_124', 'b', 
               'beta', 'cos', 'd', 'delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 
               'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log',
               'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm','prime', 'q', 'rightarrow', 'sigma', 'sin', 'sqrt',
               'sum', 'tan', 'theta', 'times', 'u', 'v', 'w', 'y', 'z', '{', '}']


# In[10]:


def recognize(img,exp,model,class_names):
#     img = cv2.imread("C:\\Users\sathw\Documents\Image segmentation\sample.jpg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_to_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_to_tensor = tf.image.resize(img_to_tensor, [45, 45])

    print(type(img_to_tensor))
    input_image = input_tensor = tf.expand_dims(img_to_tensor, axis=0)
    
    pred=model.predict(input_tensor)

#     plt.figure(figsize=(7, 9))
    i=0
    # ax = plt.subplot(10, 10, i + 2)
    print(np.argmax(pred))
    # plt.title(class_names[np.argmax(pred)])
#     s=s+class_names[np.argmax(pred)]
    print(pred)
    # plt.imshow(img)
#     input_tensor = tf.squeeze(input_tensor, axis=None, name=None)
#     print("squeezed shape ",input_tensor.shape)
#     plt.imshow(input_tensor)
    i=i+1
    # plt.axis("off")
    # plt.figure(figsize=(7, 9))
    return class_names[np.argmax(pred)]
# print(ip)


# In[11]:


def conv_exp(equations):
    for equation,i in zip(equations,range(len(equations))):
        if 'o' in equation:
            equation=equation.replace("o","0")
            equations[i]=equation
        elif "times" in equation:
            equation=equation.replace("times","*")
            equations[i]=equation
        elif ","  in equation:
            equation=equation.replace(",","/")
            equations[i]=equation
            
        print(equations)
        new_equations=[]
        for equation in equations:
            for ch in range(0,len(equation)-1):
                if equation[ch].isnumeric() and equation[ch+1].isalpha():
                    print("equation[0:ch] ",equation[0:ch+1]," ",equation[ch+1:])
                    equation=equation[0:ch+1]+"*"+equation[ch+1:]
                    print(equation)
            new_equations.append(equation)
    print(equations)
    print(new_equations)
    equations=new_equations
    return is_system_eq(equations)
    


# In[12]:


def is_system_eq(equations):
    if len(equations)>1:
        return solve_system(equations)
    else:
        return solve_single(equations)
        


# In[13]:


def solve_system(equations):
    symbols=[]
    for ch in equations[0]:
        if ch.isalpha():
            symbols.append(sympy.Symbol(ch))

    con_expression=[]
    for equation in equations:
        lhs =  sympify(equation.split("=")[0])
        rhs =  sympify(equation.split("=")[1])
        con_expression.append(lhs-rhs)

    # lhs =  sympify(string_.split("=")[0])
    # rhs =  sympify(string_.split("=")[1])
    solution = sympy.solve(con_expression,symbols)
    print(solution)
    
    final_result=solution
    return equations,solution


# In[14]:


def solve_single(equations):
    if "=" in equations[0]:
        symbols=[]
        for ch in equations[0]:
            if ch.isalpha():
                symbols.append(sympy.Symbol(ch))
        con_expression=[]
        for equation in equations:
            lhs =  sympify(equation.split("=")[0])
            rhs =  sympify(equation.split("=")[1])
            con_expression.append(lhs-rhs)
        solution = sympy.solve(con_expression,symbols)
        
    else:
        for ch,i in zip(equations[0],range(len(equations[0]))):
            if ch=="X" and equations[0][i-1]=="*":
                equations[0] = equations[0].replace("*X","*")
            elif ch=="l":
                equations[0] = equations[0].replace("l","/")
            elif ch=="N":
                equations[0] = equations[0].replace("N","0")
        solution=sympy.sympify(equations[0])
    print(solution)
    final_result=solution
    return equations,solution

# def ret_solu():
    # return final_result


# In[17]:


# read_image("C:\\Users\sathw\Documents\Image segmentation\\three.jpg")


# In[ ]:




