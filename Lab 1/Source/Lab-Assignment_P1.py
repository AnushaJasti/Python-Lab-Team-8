a = input("Enter your account number: ")
x = 0 #Initial Account Balance
n = 0
while n < 100:
    y = input("Enter your option") #Enter your option Deposit or Withdraw
    a,b = y.split(" ")
    if (a == "Deposit" or a == "Withdraw"):
        if a == "Deposit":
            t = x + int(b)
            x = t
        elif a == "Withdraw":
            t = x - int(b)
            x = t
        else:
            print("Invalid Options") #Invalid Option
    else:
        print("Transaction Completed")
        n = 1000
print("Total amount is: ", x)


