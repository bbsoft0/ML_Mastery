##########################################################################
# Hello Python
# Functions
# Booleans and Conditionals
# Lists
# Loops and List comprehensions
# Strings and Dictionaries
# Working with External Libraries

import numpy
my_favourite_things = [32, 'raindrops on roses', help]
planets = ['Mercury', 'Venus', 'Earth', 'Mars',
           'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planets[0]

# First 3 planets - Mercury', 'Venus', 'Earth'
planets[0:3]
planets[:3]

# All the planets except the first and last
planets[1:-1]

# The last 3 planets
planets[-3:]

planets.append('Pluto')

print(sorted(planets))
planets.index('Earth')

print("Pluto is at index:"+str(planets.index('Pluto')))


def mod_5(x):
    """Return the remainder of x after dividing by 5"""
    return x % 5


print(
    'Which number is biggest?',
    max(100, 51, 14),
    'Which number is the biggest modulo 5?',
    max(100, 51, 14, key=mod_5),
    sep='\n',
)

for i in range(5):
    print("Doing important work. i =", i)

squares = [n**2 for n in range(10)]
print(squares)


def count_negatives(nums):
    return len([num for num in nums if num < 0])


def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])


datestr = '1956-01-31'
year, month, day = datestr.split('-')
print(day, month, year, sep=' ')
print('/'.join([month, day, year]))

numbers = {'one': 1, 'two': 2, 'three': 3}
numbers['eleven'] = 11

planets = ['Mercury', 'Venus', 'Earth', 'Mars',
           'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))


print("numpy.random is a", type(numpy.random))
print("it contains names such as...",
      dir(numpy.random)[-15:]
      )

# Roll 10 dice
rolls = numpy.random.randint(low=1, high=6, size=10)
print(rolls)

xlist = [[1, 2, 3], [2, 4, 6], ]
# Create a 2-dimensional array
x = numpy.asarray(xlist)
print("xlist = {}\nx =\n{}".format(xlist, x))
