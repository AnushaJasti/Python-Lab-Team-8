class Person(): #Defining person class
    def __init__(self,name,age,phone_number):
        self.name = name
        self.age = age
        self.phone_number = phone_number

        self._gender = "not_metioned"
        print("Person instance is created")

    def get_details(self):
        print(f"Name - {self.name}\nAge - {self.age}\nPhone Number - {self.phone_number}")

    def set_person_gender(self, gender):
            self._gender = gender

    def get_person_gender(self):
        return self._gender

class Flight(): #Defining flight class

    def __init__(self, flight_name,flight_number):
        self.flight_name = flight_name
        self.flight_number = flight_number
        print("Flight instance  created")

class Employee(Person): #Defining Employee class by inheriting person as superclass

    def __init__(self,name,age,phone_number,employeeID):
        Person.__init__(self,name,age,phone_number)
        self.employeeID = employeeID
        print("Employee instance created")
    def print_passenger_details(self):
        super.details()

class Ticket(Flight,Person): #Defining Ticket class by inheriting Person and Flight as asuper class

    def __init__(self,ticket_number,departure_date):
        self.ticket_number = ticket_number
        self.departure_date = departure_date

class Passenger(Person,Flight): #Defining Passenger class by inheriting Person and Flight as asuper class

    def __init__(self,name,age,phone_number,luggage_weight):
        Person.__init__(self, name, age, phone_number)
        self.luggage_weight = luggage_weight

        self.__passenger_passport_number = "not updated"

    def set_passport_number(self,passport_number):
        self.__passenger_passport_number = passport_number

    def get_passport_number(self):
        return self.__passenger_passport_number

person1 = Person('Rahul',23,1234567890) #Instantiating person1
person1.set_person_gender('female')
print(person1.get_person_gender())
print(person1._gender)
employee1 = Employee("Rupesh",22,1234567890,'14b81a12b6') #Instantiating employee1
employee2 = Employee("Anusha",21,46178918291,'14b81a12c0') #Instantiating employee2
employee1.get_details()
passenger1 = Passenger("Nikhil",21,1234567890,67) #Instantiating person1
passenger1.set_passport_number('AA123456')
print(passenger1.get_passport_number())