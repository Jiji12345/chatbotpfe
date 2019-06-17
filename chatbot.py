#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer
from autocorrect import spell

import os
from six.moves import cPickle
import re


# In[2]:


from tkinter import *
from tkinter import messagebox


# In[3]:


MAX_LEN = 25
BATCH_SIZE = 64

stemmer = PorterStemmer()
def process_str(string, bot_input=False, bot_output=False):
    string = string.strip().lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`:]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.split(" ")
    string = [re.sub(r"[0-9]+", "NUM", token) for token in string]
    string = [stemmer.stem(re.sub(r'(.)\1+', r'\1\1', token)) for token in string]
    string = [spell(token).lower() for token in string]
    # Truncate string
    while True:
        try:
            string.remove("")
        except:
            break
    if(not bot_input and not bot_output):
        string = string[0:MAX_LEN]
    elif(bot_input):
        string = string[0:MAX_LEN-1]
        string.insert(0, "</start>")
    else:
        string = string[0:MAX_LEN-1]
        string.insert(len(string), "</end>")
    old_len = len(string)
    for i in range((MAX_LEN) - len(string)):
        string.append(" </pad> ")
    string = re.sub("\s+", " ", " ".join(string)).strip()
    return string, old_len


# In[4]:


imported_graph = tf.train.import_meta_graph('checkpoints/best_validation.meta')
sess = tf.InteractiveSession()
imported_graph.restore(sess, "checkpoints/best_validation")

sess.run(tf.tables_initializer())
graph = tf.get_default_graph()


# In[5]:


def test(text):
    text, text_len = process_str(text)
    text = [text] + ["hi"] * (BATCH_SIZE-1)
    text_len = [text_len] + [1] * (BATCH_SIZE-1)
    return text, text_len


# In[6]:


test_init_op = graph.get_operation_by_name('data/dataset_init')

user_ph = graph.get_tensor_by_name("user_placeholder:0")
bot_inp_ph = graph.get_tensor_by_name("bot_inp_placeholder:0")
bot_out_ph = graph.get_tensor_by_name("bot_out_placeholder:0")

user_lens_ph = graph.get_tensor_by_name("user_len_placeholder:0")
bot_inp_lens_ph = graph.get_tensor_by_name("bot_inp_lens_placeholder:0")
bot_out_lens_ph = graph.get_tensor_by_name("bot_out_lens_placeholder:0")

words = graph.get_tensor_by_name("inference/words:0")


# In[7]:


def chat(text):
    user, user_lens = test(text)
    sess.run(test_init_op, feed_dict={
        user_ph: user,
        bot_inp_ph: ["hi"] * BATCH_SIZE,
        bot_out_ph: ["hi"] * BATCH_SIZE,
        user_lens_ph: user_lens,
        bot_inp_lens_ph: [1] * BATCH_SIZE,
        bot_out_lens_ph: [1] * BATCH_SIZE
    })
    translations_text = sess.run(words)
    output = [item.decode() for item in translations_text[0]]
    if("</end>" in output):
        end_idx = output.index("</end>")
        output = output[0:end_idx]
    output = " ".join(output)
    return(output)


# In[8]:


def listen():
    utterance=input()
    return (utterance)


# In[9]:


def Enter_Hit(event):

    utterance=input_user.get()
    message.config(state=NORMAL,fg="Black")
    message.insert("end","  You : "+utterance+"\n ")
    message.see("end")
    
    resp = chat(utterance)
    message.config(state=NORMAL,fg="Blue")
    message.insert("end", " Chatbot : " +resp+" \n \n","bot")
    message.see("end")
    message.config(state=DISABLED)


# In[10]:


def on_closing():

    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        GUI.destroy()


# In[11]:


GUI=Tk()
GUI.title("Assistant Chatbot")
scrollbar = Scrollbar(GUI)
scrollbar.pack(side=RIGHT, fill=Y)
label1=Label(GUI,text="Welcome  !",fg="Red")
label1.pack(side=TOP,fill=X)
positionRight = int(GUI.winfo_screenwidth()/2 - GUI.winfo_reqwidth()/2)
positionDown = int(GUI.winfo_screenheight()/2 - GUI.winfo_reqheight()/2)
w = int(positionRight - GUI.winfo_reqwidth())
h = int(positionDown - GUI.winfo_reqheight())
message=Text(GUI)
message.config(state=NORMAL,fg="Blue")
message.insert("end", " Chatbot : I'm your virtual assistant , you can start chatting with me now ! \n  \n" )
message.pack()
message.config(state=DISABLED,yscrollcommand=scrollbar.set )
scrollbar.config(command=message.yview)
input_user=Entry(GUI,text=input)
user_label=Label(GUI,text="Enter message here:",fg="Red")
user_label.pack(fill=X)
input_user.pack(fill=X)
input_user.bind("<Return>", Enter_Hit)
GUI.protocol("WM_DELETE_WINDOW", on_closing)
GUI.geometry("+{}+{}".format(w, h))
GUI.mainloop()





