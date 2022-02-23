##########################################################################
# Expected Value

from math import comb
##########################################################################
# Loto 6/49
print(f"Loto 6/49")
# Romania
prize_money = {0: 0, 1: 0, 2: 0, 3: 30, 4: 238, 5: 15837, 6: 12700000}
# When jackpot hits 5 Million or above.
#prize_money = {0: 0, 1: 0, 2: 0, 3: 50, 4: 1000, 5: 25000, 6: 5000000 }

# Probability of getting n blue ball ?
expected_win_prize = 0
matches = []
prizes = []
for i in range(0, 7):
    w_prob = (comb(43, 6-i) * comb(6, i)) / comb(49, 6)
    # n!/k!(n-k)!
    desc_prob = comb(49, 6) / (comb(43, 6-i) * comb(6, i))

    #total+= w_prob
    print(f"Probability of getting {i} blue ball = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money[i]
    matches.append((i, w_prob))
    prizes.append((prize_money[i], w_prob))

print(
    f"Expected winning prize of a single lottery is : {round(expected_win_prize,4)}")
print(f"Expected Value  is : {round(expected_win_prize-5,4)}")


##########################################################################
# Loto 5/40
print(f"Loto 5/40")
# Romania
prize_money = {0: 0, 1: 0, 2: 0, 3: 0, 4: 405, 5: 50000, 6: 1170000}
# When jackpot hits 5 Million or above.
#prize_money = {0: 0, 1: 0, 2: 0, 3: 50, 4: 1000, 5: 25000, 6: 5000000 }

# Probability of getting n blue ball ?
expected_win_prize = 0
matches = []
prizes = []
for i in range(0, 7):
    w_prob = (comb(43, 6-i) * comb(6, i)) / comb(49, 6)
    # n!/k!(n-k)!
    desc_prob = comb(49, 6) / (comb(43, 6-i) * comb(6, i))

    #total+= w_prob
    print(f"Probability of getting {i} blue ball = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money[i]
    matches.append((i, w_prob))
    prizes.append((prize_money[i], w_prob))

print(
    f"Expected winning prize of a single lottery is : {round(expected_win_prize,4)}")
print(f"Expected Value  is : {round(expected_win_prize-5,4)}")

##########################################################################
# Joker
print(f"===Joker================================")
#prize_money = {0: 37, 1: 28, 2: 360, 3: 410, 4: 26250, 5: 52500, 6: 29151000}
# When jackpot hits 5 Million or above.
prize_money = {0: 0, 1: 0, 2: 0, 3: 108, 4: 3320, 5: 288500}
prize_money_joker = {0: 0, 1: 17, 2: 40,
                     3: 670, 4: 66250, 5: 29100000, }

expected_win_prize = 0
prizes = []
lose_prob = 0
joker_prob = 20
for i in range(0, 6):
    # n!/k!(n-k)!
    orig_w_prob = (comb(40, 5-i) * comb(5, i)) / comb(45, 5)
    orig_desc_prob = comb(45, 5) / (comb(40, 5-i) * comb(5, i))

    # No Joker
    w_prob = orig_w_prob * (joker_prob-1)/joker_prob
    desc_prob = orig_desc_prob * (joker_prob)/(joker_prob-1)
    print(f"Probability of getting {i} blue ball = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money[i]
    prizes.append((prize_money[i], w_prob))
    if (int(prize_money[i]) == 0):
        lose_prob += w_prob
   # Yes Joker
    w_prob = orig_w_prob / joker_prob
    desc_prob = orig_desc_prob * joker_prob
    print(
        f"Probability of getting {i} blue ball + Joker = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money_joker[i]
    prizes.append((prize_money_joker[i], w_prob))
    if (int(prize_money_joker[i]) == 0):
        lose_prob += w_prob

print(
    f"Expected winning prize of a single lottery is : {round(expected_win_prize,4)} ron")
print(
    f"Expected Value  is : {round(expected_win_prize-4*lose_prob,4)}    {round((expected_win_prize-4*lose_prob)*100/4,2)} %")
print(
    f"Probability to win nothing on a single ticket : {round(lose_prob*100,2)} %")

##########################################################################
# Powerball
print(f"===Powerball================================")
#prize_money = {0: 37, 1: 28, 2: 360, 3: 410, 4: 26250, 5: 52500, 6: 28451000}
# When jackpot hits 5 Million or above.
prize_money = {0: 0, 1: 0, 2: 0, 3: 7, 4: 100, 5: 1000000}
prize_money_joker = {0: 4, 1: 4, 2: 7,
                     3: 100, 4: 50000, 5: 236000000}

expected_win_prize = 0
prizes = []
lose_prob = 0
joker_prob = 26
for i in range(0, 6):
    # n!/k!(n-k)!
    orig_w_prob = (comb(64, 5-i) * comb(5, i)) / comb(69, 5)
    orig_desc_prob = comb(69, 5) / (comb(64, 5-i) * comb(5, i))

    # No Bonus
    w_prob = orig_w_prob * (joker_prob-1)/joker_prob
    desc_prob = orig_desc_prob * (joker_prob)/(joker_prob-1)
    print(f"Probability of getting {i} blue ball = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money[i]
    prizes.append((prize_money[i], w_prob))
    if (int(prize_money[i]) == 0):
        lose_prob += w_prob
   # Yes Bonus
    w_prob = orig_w_prob / joker_prob
    desc_prob = orig_desc_prob * joker_prob
    print(
        f"Probability of getting {i} blue ball + Joker = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money_joker[i]
    prizes.append((prize_money_joker[i], w_prob))
    if (int(prize_money_joker[i]) == 0):
        lose_prob += w_prob

print(
    f"Expected winning prize of a single lottery is :  ${round(expected_win_prize,4)} ")
print(
    f"Expected Value  is : {round(expected_win_prize-2*lose_prob,4)}    {round((expected_win_prize-2*lose_prob)*100/2,2)} %")
print(
    f"Probability to win nothing on a single ticket : {round(lose_prob*100,2)} %")

##########################################################################
# MegaMillions
print(f"===MegaMillions================================")
#prize_money = {0: 37, 1: 28, 2: 360, 3: 410, 4: 26250, 5: 52500, 6: 28451000}
# When jackpot hits 5 Million or above.
prize_money = {0: 0, 1: 0, 2: 0, 3: 10, 4: 500, 5: 1000000}
prize_money_joker = {0: 2, 1: 4, 2: 10,
                     3: 200, 4: 10000, 5: 515000000, }

expected_win_prize = 0
prizes = []
lose_prob = 0
joker_prob = 25
for i in range(0, 6):
    # n!/k!(n-k)!
    orig_w_prob = (comb(65, 5-i) * comb(5, i)) / comb(70, 5)
    orig_desc_prob = comb(70, 5) / (comb(65, 5-i) * comb(5, i))

    # No Bonus
    w_prob = orig_w_prob * (joker_prob-1)/joker_prob
    desc_prob = orig_desc_prob * (joker_prob)/(joker_prob-1)
    print(f"Probability of getting {i} blue ball = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money[i]
    prizes.append((prize_money[i], w_prob))
    if (int(prize_money[i]) == 0):
        lose_prob += w_prob
   # Yes Bonus
    w_prob = orig_w_prob / joker_prob
    desc_prob = orig_desc_prob * joker_prob
    print(
        f"Probability of getting {i} blue ball + Joker = {round(desc_prob,2)}")
    expected_win_prize += w_prob * prize_money_joker[i]
    prizes.append((prize_money_joker[i], w_prob))
    if (int(prize_money_joker[i]) == 0):
        lose_prob += w_prob

print(
    f"Expected winning prize of a single lottery is : ${round(expected_win_prize,4)} ")
print(
    f"Expected Value  is : {round(expected_win_prize-2*lose_prob,4)}    {round((expected_win_prize-2*lose_prob)*100/2,2)} %")
print(
    f"Probability to win nothing on a single ticket : {round(lose_prob*100,2)} %")
