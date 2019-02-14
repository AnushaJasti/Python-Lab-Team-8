list=[]
tupdict={}
tup=()
tup1=()
listtup=()
list1=[]
num=int(input("Enter number of tuples: ")) #Enter no. of tuples you want to enter
for i in range(num):
    tup = ()
    tup1 = ()
    listtup = ()
    name=input("Enter the name: ") #Enter your name
    temp=tuple([name])
    sub=input("Enter subject name: ") #Enter subject name
    temp1=tuple([sub])
    marks=input("Enter marks for that subject: ") #Enter marks
    temp2=tuple([marks])
    tup1=temp1+temp2
    listtup=temp,tup1
    list.append(listtup) #Appending to the list
    del tup
    del tup1
    del temp
    del temp1
    del temp2
    del listtup #Deleting all the tuples
for i in list:
    tup= i
    if tup[0] in tupdict:
        tup1= tup[1]+tupdict[tup[0]]
        tupdict[tup[0]]=tup1
    else:

        tupdict.update({tup[0]:tup[1]})
print(tupdict)