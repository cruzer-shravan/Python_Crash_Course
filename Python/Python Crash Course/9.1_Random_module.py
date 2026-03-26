"""Python standard library is a set of modules included with every Python installation."""
# You can use any function or class in the standard library 
# by including a simple import statement at the top of your file.

# one module, RANDOM, which can be useful in modeling many real-world situations.

from random import randint, random
# print(randint(1,6))

# from random import choice
# players = ['charles', 'martina', 'michael', 'florence', 'eli']
# first_up = choice(players)
# print(first_up)

# random module shouldn't be used when building security-related applications.
# but it works well for many fun and interesting projects.

# '''You can also download modules from external sources.'''

# """TRY IT YOURSELF"""

# # 9.13. DICE

# class Die:
#     """This class represents a Die with 6 sides and its methods."""

#     def __init__(self, sides):
#         self.sides = sides
        
#     def roll_die(self):
#         random_die = randint(1, self.sides)
#         print(f"random num: {random_die}")

# outcome = Die(6)
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()

# outcome = Die(10)
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()

# outcome = Die(20)
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()
# outcome.roll_die()


# # 9.14. LOTTERY

# from random import choice
# # choice returns one randomly selected element from a sequence (list, tuple or string)

# # Creatae a list with 10 numbers and 5 letters
# alphanumeric = [0,1,2,3,4,5,6,7,8,9,'a','e','i','o', 'u']

# # Ramndomly select 4 numbers or letters from the list
# lottery_ticket = []

# while len(lottery_ticket) < 4:
#     picked = choice(alphanumeric)       

#     if picked not in lottery_ticket:
#         lottery_ticket.append(picked)

# print(f"Any ticket matching these 4 numbers or letters wins a prize: {lottery_ticket}")

# '''
# ⚖️ choice vs sample

# Function	Picks	Repetition	Output
# choice()	1 item	Can repeat	Single value
# sample()	Multiple items	No repetition	List
# '''

# import random

# # Randomly select 4 letters/digits from list of alphanumeric
# winning_ticket = random.sample(alphanumeric, 4)

# print("Winning ticket must match the following letters/digits in sequence:", winning_ticket)


# 9.15. LOTTERY ANALYSIS

from random import choice
import time

# Start time
# start_time =  print(f"Start time:{time.time()}")    # incorrect

# correct method
start_time =  time.time()
print(start_time)

alphanumeric = [0,1,2,3,4,5,6,7,8,9,'a','e','i','o', 'u']

my_ticket = []

while len(my_ticket) < 6:
    picked = choice(alphanumeric)

    if picked not in my_ticket:
        my_ticket.append(picked)

print(f"My lottery ticket: {my_ticket}")

# final_ticket = []

loop = 0


# GPT - OPTION 2: Reset each attempt (true lottery simulation)

while True:
    final_ticket = []
    
    while len(final_ticket) < len(my_ticket):
        picked = choice(alphanumeric)
        if picked not in final_ticket:
            final_ticket.append(picked)
    
    loop += 1
    
    if final_ticket == my_ticket:
        break


# # GPT - OPTION 3: Direct comparison (fastest)

# import random

# while True:
#     final_ticket = random.sample(alphanumeric, len(my_ticket))
#     loop += 1
    
#     if final_ticket == my_ticket:
#         break

# while final_ticket != my_ticket:      # OPTION 1: IGNORE ORDER (RECOMMENDED) - SELF CODE
#     '''
#     ⚠️ Why it's inefficient (final_ticket != my_ticket)
#     You are:
#     Randomly picking values
#     Trying to match 'exact list' order
#     Without resetting final_ticket
#     '''

#     picked = choice(alphanumeric)

#     if picked in my_ticket:
#         if picked not in final_ticket:
#             final_ticket.append(picked)
#         loop += 1                             
#     else:                                     
#         loop += 1                             
#       

# End time
# end_time = print(f"End time:{time.time()}")   # incorrect

print(f"Winning ticket: {final_ticket}")
# correct method
end_time =  time.time()
print(end_time)

print(f"Number of loops iterated to pick the winning ticket: {loop}")
print(f"Runtime: {end_time - start_time:.6f} seconds")

