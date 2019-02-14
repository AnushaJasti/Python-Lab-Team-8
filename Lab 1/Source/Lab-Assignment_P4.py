import copy #importing the copy library
sentence= input("Enter the String")
substring=[]
substring2=[]
max=0
for i in range(len(sentence)):
    if sentence[i] not in substring:
        substring.append(sentence[i]) #Appending the letter to the substring if condition satisfies
    else:
        if len(substring)>max:
            substring2.clear()
            substring2=copy.deepcopy(substring)
            substring.clear()
            substring.append(sentence[i])
            max=len(substring2)
        else:
            substring.clear()
if len(substring)>max:
    substring2.clear()
    substring2 = copy.deepcopy(substring)
    max=len(substring2)
print(substring2," ",max)

