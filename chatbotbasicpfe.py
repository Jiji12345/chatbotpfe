
# coding: utf-8

# In[1]:


from tkinter import *          
from random import choice
from tkinter import messagebox
# Load data preprocessing libs
import pandas as pd
import numpy as np
import copy
# Load vectorizer and similarity measure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


def on_closing():

    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


# In[ ]:


root = Tk()                                         
user = StringVar()                                      
bot  = StringVar()

#Label(root, text=" user : ").pack(side=LEFT)                   
#Entry(root, textvariable=user).pack(side=LEFT)     
#Label(root, text=" Bot  : ").pack(side=LEFT)        
#Entry(root, textvariable=bot).pack(side=LEFT) 

root.title("chatbot PFE")
root.configure(background='#008080')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)
message=Text(root) 
message.pack()
message.config(state=DISABLED,yscrollcommand=scrollbar.set)
scrollbar.config(command=message.yview)
user_label=Label(root,text="Enter message here:",fg="Green")
user_label.pack(fill=X)
Entry(root,textvariable=user).pack(fill=X)
user_label=Label(root,text="response:",fg="Green")
user_label.pack(fill=X)
Entry(root, textvariable=bot).pack(fill=X) 


#utterance=user.get()
#message.tag_config("bot",foreground="black")
#message.tag_config("user", foreground="blue")
#utter=copy.deepcopy(utterance)

#utter = prepare_utterance(utter)
#print(utter)
#message.config(state=NORMAL,fg="Blue")
#message.insert("end","User: "+utterance+"\n","user")
#message.see("end")
    
    
    
def main():
    df = pd.read_csv("helpd.csv")
    df.dropna(inplace=True)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(np.concatenate((df.Question, df.Answer)))
    Question_vectors = vectorizer.transform(df.Question)
    # while True:
    # Read user input
    question = user.get()

    # Locate the closest question
    input_question_vector = vectorizer.transform([question])

    # Compute similarities
    similarities = cosine_similarity(input_question_vector, Question_vectors)

    # Find the closest question
    closest = np.argmax(similarities, axis=1)

    # Print the correct answer
    bot.set(df.Answer.iloc[closest].values[0])

def com():
    c=user.get()
    label1=Label(message, text="user : "+c).pack()
    d=bot.get()
    label2=Label(message, text="bot : "+d).pack()
    
root.protocol("WM_DELETE_WINDOW", on_closing)    
Button(root, text="speak", command=main).pack(side=LEFT)
Button(root, text="print", command=com).pack(side=LEFT)
mainloop()


# In[ ]:




