#Guessing

from random import randint
magic_number=(randint(0, 10))

guess=input("Pick a number between 0 and 10 :")

if (int(guess)==magic_number):
    print ("You have WON !")
else:
    print ("Miss... Run the program again. %s",magic_number )
