python=set() #Creating empty set
web=set()
while True:
    students=str(input("Enter student name for python class")) #Enter details of student
    python.add(students) #Adding students names to python app set
    contin= input("do you wish to add more students(y/n)") #Enter your preference
    if contin.lower().__eq__('y'):
        continue
    else
        break
while True:
    student1=input("Enter student name for web application class:")
    web.add(student1) #Adding students names to web app set
    contin1= input("Do you wish to add more students(y/n): ")
    if contin1.lower().__eq__('y'):
        continue
    else:
        break
print("list of students who are in python class:", python)
print("list of students who are in web application class :",web)
print("list of students who are attending both the classes are :",python&web) #Displaying the students names who are attending both the classes
print("list of students who are not common in both the classes are: ",python ^ web) #Displaying the students names who are not common for both the classes