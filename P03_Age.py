# Age in seconds


def age_in_seconds():
    age = input("Age please:")
    intage = float(age)
    seconds = intage*365*24*60*60
    return seconds


def run():
    age = age_in_seconds()
    print("Your age in seconds is: {}".format(age))


run()
