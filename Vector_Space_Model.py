from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image
from collections import Counter
import math



class GUI(Tk):
    def __init__(self): 
        Tk.__init__(self)
        self.tokken=[]
        self.tag=[]  
        self.inverted=dict()
        self.tokkens=[]
        self.term=[]
        self.idf=dict()        
        self.tf_idf=dict()
        self.magnitude=[]
        
        self.preprocess()
        self.gui_generate()
    
    def gui_generate(self):
        self.image=ImageTk.PhotoImage(file="vector1.1.jpg")
        self.label=Label(self, image=self.image)
        self.label.pack(fill=BOTH,expand=True)

        self.l=Label(self,text="Vector Space Model", bg="navy", fg="white" ,font="Courier 30 bold")
        # self.la=Label(self,text='', bg="black", fg="white",font="Courier 10", wraplength=400)
        self.entry=Entry(self,width=70,bd=3,font="Times_New_Roman 12")
        self.entry.bind("<Return>",self.queryProcess)
        self.f=Frame(self)
        self.b1=Button(self,text="Search",bg="navy", bd=3,fg="white",font="Courier 15 bold")
        self.b1.bind("<Button-1>",self.queryProcess)

        self.b=Button(self,text="Exit",command=exit,bd=3,bg="navy", fg="white",font="Courier 15 bold")
        self.v = Scrollbar(self.f)
        self.v.pack(side = RIGHT, fill = Y)
        self.t = Text(self.f,width = 100, height = 15, fg="white",bg="navy",font="Cambria 10 bold",yscrollcommand = self.v.set)
        
        self.l.place(relx=0.29, rely=0.15,height=50)
        self.entry.place(relx=0.17, rely=0.35, height=35)
        self.b1.place(relx=0.73, rely=0.35)
        self.b.place(relx=0.92, rely=0.9)
        # self.la.place(relx=0,rely=0)
        self.f.place(relx=0,rely=0.5)
        self.f.place_forget()
        self.t.pack(side=TOP)
        self.v.config(command=self.t.yview)

    
    def preprocess(self):
        self.lemmatizer=WordNetLemmatizer()
        self.stopword=open("Stopword-List.txt","r",encoding='utf-8')
        self.stopword=self.stopword.read()

        for i in range(1,51):
            # creating tokkens
            self.f=open("ShortStories/"+str(i)+ ".txt","r",encoding='utf-8')
            self.line=self.f.read().replace(","," ").replace("."," ").replace("?"," ").replace("!"," ").replace("“"," ").replace("”"," ").replace(";"," ").replace(":"," ").replace("’"," ").replace("‘"," ").replace("—"," ").replace("-"," ").replace("n’t","not").replace("\n"," ").replace("("," ").replace(")"," ").replace("*"," ").replace("["," ").replace("]"," ").lower().split()
            self.tokken.append(self.line)
            self.f.close()
            
            self.f=open("ShortStories/"+str(i)+ ".txt","r",encoding='utf-8')
            self.tag.append(self.f.readline().replace("\n",""))
            self.f.close()
            

        for i in range(0,50):  
            l=[]
            # creating lemmatize tokkens with removal of stopword
            for letter in self.tokken[i]:
                k=self.lemmatizer.lemmatize(letter)
                l.append(k)
                if k not in self.tokkens and letter not in self.stopword:
                    self.tokkens.append(k)
            self.tokken[i]=l

        for i in range(0,50):
            # term contain frequency of each term in each document
            self.term.append(Counter(self.tokken[i]))
        
        # making inverted index
        for i in range(len(self.tokken)):
            for j in self.tokken[i]:
                index=[]
                if j in self.inverted:
                    index=self.inverted[j]
                if i+1 not in index:
                    index.append(i+1)
                self.inverted[j]=index
        
        # calculating idf by the formula N/log(df)  where len(inverted[i]) gives the number of docs single term appears
        for i in self.tokkens:
            self.idf[i]=math.log10(len(self.inverted[i]))/50

        # calculating tf_idf of each document by multiplying frequency of each term with idf
        for i in self.tokkens:
            k={}
            for j in range(0,50):
                k[j]=self.idf[i]*self.term[j][i]
            self.tf_idf[i]=k

        count=0
        # calculating magnitude for unit vector conversion
        for i in self.tokkens:
            for j in range(0,50):
                if count<1:
                    self.magnitude.append(abs(self.tf_idf[i][j]**2))
                else:
                    self.magnitude[j]+=abs(self.tf_idf[i][j]**2)
            count+=1    


    def queryProcess(self,event):
        query=self.entry.get()
        query=query.split()
        q={}
        result={}
        mag=0
        for i in self.tokkens:
            q[i]=0

        # making a lemmatize query vector
        for i in query:
            if i not in self.stopword:
                q[self.lemmatizer.lemmatize(i)]+=1
        # calculating tf_idf of q and magnitude of query vector for unit vector
        for i in self.idf:
            q[i]*=self.idf[i]
            mag+=q[i]**2
            
        for i in range(1,51):
            result[i]=0

        # computing formula result=d*q
        for i in self.tokkens:
            for j in range(0,50):
                result[j+1]+=abs(self.tf_idf[i][j])*abs(q[i])
        # computing formula result=result/|d|*|q|
        for i in range(0,50):
            result[i+1]=result[i+1]/(math.sqrt(self.magnitude[i])*math.sqrt(mag))
        
        sett=[]
        # making a set of docs whose cosing similarity is greater than 0.005
        for i in range(1,51):
            if result[i]>0.005:
                sett.append(i)

        if len(sett)>0:
            name=dict()
            for i in sett:
                name[i]=self.tag[i-1]
            self.f.place(relx=0,rely=0.5)
            self.t.delete('1.0', END)
            self.t.insert(END,"Length: "+str(len(sett))+"\n")
            for i,j in name.items():
                self.t.insert(END,"Document#"+str(i)+"    "+"Story: "+str(j))
                self.t.insert(END,"\n")
        else:
            self.t.delete('1.0', END)
            self.t.insert(END,"No results found \n")
        






root=GUI()
root.title("Model")
root.geometry("1080x500")
root.mainloop()
