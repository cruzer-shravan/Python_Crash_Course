# '''Reading from a file'''

# '''Reading the contents of a file'''

# from pathlib import Path

# path = Path('D:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\pi_digits.txt')
# contents = path.read_text()
# print(contents)

# contents = contents.rstrip()
# print(contents)

# contents = path.read_text().rstrip()
# print(contents)


# '''RELATIVE AND ABSOLUTE FILE PATHS'''

# # To get Python to # open files from a directory other than the one where your program file is stored, 
# # You need to provide the correct path.
# # There are two main ways to specify paths in programming.

# # path = Path('text_files/filename.txt')

# # You can use an absolute path if a relative path doesn’t work.
# # You’ll need to write out an absolute path to clarify where you want Python to look.

# # path = Path('D:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\pi_digits.txt')

# '''ACCESSING A FILE'S LINES'''

# '''
# For example, you might
# want to read through a file of weather data and work with
# any line that includes the word sunny in the description of
# that day’s weather. In a news report, you might look for any
# line with the tag <headline> and rewrite that line with a
# specific kind of formatting.

# You can use the splitlines() method to turn a long string
# into a set of lines, and then use a for loop to examine each
# line from a file, one at a time:
# '''

# # from pathlib import Path
# # path = Path('pi_digits.txt')
# contents = path.read_text()
# lines = contents.splitlines()
# for line in lines:
#     print(line)


# '''WORKING WITH A FILE'S CONTENTS'''

# contents = path.read_text()
# lines = contents.splitlines()
# pi_string = ''
# for line in lines:
#     pi_string += line

# print(pi_string)
# print(len(pi_string))

# '''using lstrip function'''

# for line in lines:
#     pi_string += line.lstrip()

# print(pi_string)
# print(f"Len after lstrip: {len(pi_string)}")

# print(type(pi_string))

# pi_string = pi_string.replace(" ", "").replace(".", "")
# pi_string = int(pi_string)
# print(type(pi_string))

# pi_string = float(pi_string)
# print(type(pi_string))

# '''LARGE FILES: ONE MILLION DIGITS'''

# path = Path('D:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\pi_million_digits.txt')
# contents = path.read_text()

# lines = contents.splitlines()
# pi_string = ''
# for line in lines:
#     pi_string += line.lstrip()

# print(f"{pi_string[:52]}....")
# print(len(pi_string))


# '''IS YOUR BIRTHDAY CONTAINED IN PI?'''

# # find out if someone’s birthday appears anywhere in # the first million digits of pi.
# # We can do this by expressing each birthday as a string of digits and seeing if that string appears anywhere in pi_string.

# birthday = input("Enter your birthday, in the form mmddyy: ")
# if birthday in pi_string:
#     print("Your birthday appears in the first million digits of Pi!")
# else:
#     print("Your birthday does not appear in the first million digits of Pi!")



# """TRY IT YOURSELF"""

# # 10.1. LEARNING PYTHON

# from pathlib import Path

# path = Path("D:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\learning_python.txt")
# contents = path.read_text()
# print(contents)

# lines = contents.splitlines()
# # for line in lines:
# #     print(line)

# learn_python = ''

# for line in lines:
#     learn_python += line

# print(learn_python)


# # 10.2. learning C

# learn_python = learn_python.replace('python', 'java')
# print(learn_python)

# 10.3. Simpler code

# for line in contents.splitlines():
#     print(line)


'''WRITING TO A FILE - Simplest way to save data is to write it to a file'''

'WRIITING A SINGLE LINE'

# Once you have a path defined, you can write to a file using the write_text() method.

from pathlib import Path
# path = Path('programming.txt')
# path.write_text("I love programming.")

# print(path.resolve())

# import os
# print(os.getcwd())

# base_dir = Path(__file__).parent
# path = base_dir / "programming.txt"

# path.write_text("I am enjoying programming")

# # NOTE: Python can only write strings to a text file.


# 'WRITING MULTIPLE LINES'

# # The write_text() method does a few things behind the scenes.
# # If the file that path points to doesn’t exist, it creates that file. 
# # Also, after writing the string to the file, it makes sure the file is closed properly. 
# # Files that aren’t closed properly can lead to missing or corrupted data.

# # Writing several line to the programming.txt

# contents = "I love creating new games.\n"
# contents += "I also love working with data.\n"  # There is no limit to the length of your strings.

# path = base_dir / "programming.txt"
# path.write_text(contents)       # Adding the contents is actually deleting the existing content in the .txt file

# # NOTE
# # Be careful when calling write_text() on a path object.
# # If the file already exists, write_text() will erase the
# # current contents of the file and write new contents to the file.

# '''
# 💡 Important difference
#   Method	                Behavior
# write_text()	        ❌ Overwrites file
# open(..., "w")	    ❌ Overwrites file
# open(..., "a")	    ✅ Appends to file

#  "w" → 🧹 Erase board, then write
# "a" → ✍️ Keep writing on the board
# '''


# '''TRY IT YOURSELF'''

# # 10.4. Guest / 10.5. Guest book

# from pathlib import Path

# base_dir = Path(__file__).parent
# path = base_dir / "guest.txt"

# while True:
#     print("Enter your name or type 'q' to exit!")
#     user_name = input("Enter your name: ").lower()
    
#     if user_name == 'q':
#         break
#     else:
#         with open(path, "a") as file:
#             file.write(f"{user_name}\n")


# '''EXCEPTIONS'''

# # Python uses special objects called exceptions to manage errors that arise during a program’s execution.
# # If you write code that handles the exception, the program will continue running.
# # If you don’t handle the exception, the program will halt and show a traceback,
# # which includes a report of the exception that was raised.

# # Exceptions are handled with try-except blocks. 
# # A try-except block asks Python to do something, 
# # but it also tells Python what to do if an exception is raised.

# '''Handling the ZeroDivisionError (Division by zero) Exception'''

# # You probably know that it’s impossible to divide a number by zero,
# # but let’s ask Python to do it anyway.

# 'print(5/0)'

# # The above print code reports a traceback error, ZeroDivisionError, is an exception object.
# # We can use this information to modify our program.
# # We’ll tell Python what to do when this kind of exception occurs


# '''Using try-except Blocks'''

# # When you think an error may occur, you can write a tryexcept block to handle the exception that might be raised.
# # try-except Block tells Python what to do if the code results in a particular kind of exception.

# 'try-except block for handling the ZeroDivisonError'

# try:
#     print(5/0)
# except ZeroDivisionError:
#     print("You can't divide by zero!")  # user sees a friendly error message instead of a traceback

# # If the code in the try block causes an error, Python looks for an except block whose error matches the
# # one that was raised, and runs the code in that block.

# # If more code followed the try-except block, 
# # the program would continue running because we told Python how to handle the error.


# '''Using Exveptions to prevent crashes'''

# # Simple Calculator that does only division:

# print("Give me two numbers and I'll divide them.")
# print("Enter 'q' to quit.")

# while True:
#     first_num = input("\nFirst number: ")
#     if first_num == 'q':
#         break
#     second_num = input("\nSecond number: ")
#     if second_num == 'q':
#         break
#     answer = int(first_num)/ int(second_num)
#     print(f"Result: {answer}")

#     try_again = input("Enter 'y' to try again: ").lower()
#     if try_again != 'y':
#         print("Thank you. See you soon!")
#         break


# It’s bad that the program crashed (ZeroDivisionError),
# but it’s also not a good idea to let users see tracebacks.
# Nontechnical users will be confused by them, 
# and in a malicious setting, attackers will learn more than you want them to.

# A skilled attacker can sometimes use this information to determine which kind
# of attacks to use against your code.


# '''The else block'''

# # We can make this program more error resistant by wrapping
# # the line that might produce errors in a try-except block.

# # Any code that depends on the try block executing successfully goes in the else block

# while True:
#     first_num = input("\nFirst number: ")
#     if first_num == 'q':
#         break
#     second_num = input("\nSecond number: ")
#     if second_num == 'q':
#         break
    
#     # The only code that should go in a try block is the code that
#     # might cause an exception to be raised.
    
#     try:                
#         answer = int(first_num)/ int(second_num)
#     except ZeroDivisionError:
#         print("You can't divide by a number by 0")
#     else:
#         print(f"Result: {answer}")

#     try_again = input("Enter 'y' to try again: ").lower()
#     if try_again != 'y':
#         print("Thank you. See you soon!")
#         break


# '''Handling the FileNotFoundError Exception'''

# One common issue when working with files is handling missing files
# (different location, incorrect filename or the file might not exist at all.)

# from pathlib import Path
# path = Path('alice.txt')    # file not saved in cwd

# contents = path.read_text(encoding='utf-8')

# we’re using read_text() in a slightly different way.
# encoding argument is needed when your system’s default encoding doesn’t match
# the encoding of the file that’s being read.
# reading from a file that wasn’t created on your system.

# Python can’t read from a missing file, so it raises an exception: 'FileNotFoundError'

# It’s often best to start at the very end of the complex traceback.
# This is important because it tells us what kind of exception to use in the except block that we’ll write.


# try:
#    contents = path.read_text(encoding="utf-8") 
# except FileNotFoundError:
#    print(f"Sorry, the file {path} does not exist.")

# the code in the try block produces a FileNotFoundError,
# so we write an except block that matches that error


'''ANALYZING TEXT'''

# You can analyze text files containing entire books.
# Project Gutenberg (https://gutenberg.org) maintains a collection of literary works that are available in the public domain,
# and it’s a great resource if you’re interested in working with literary texts in your programming projects.


# from pathlib import Path

# base_dir = Path(__file__).parent
# path = base_dir / "txt files"/ "alice.txt"   # modified after moving the files to new folder

# try:
#     contents = path.read_text(encoding="utf-8")
# except FileNotFoundError:
#     print(f"Sorry, the file {path.stem} does not exist.")
# else:                                                    
#     # else block works only if try block is executed sucessfully
#     # Count the approximate number of words in the file
#     words = contents.split()
#     num_words = len(words)
#     print(f"The file {path.name} has about {num_words} words.")

# '''
# 🧠 Key Path attributes (very useful)
# path.name → alice.txt ✅ (what you want)
# path.stem → alice (without extension)
# path.suffix → .txt
# path.parent → directory path
# '''

# '''WORKING WITH MULTIPLE FILES'''

# # Creating a def function "count_words()" will make it easier to run analysis on multiple books.

# from pathlib import Path
# base_dir = Path(__file__).parent

# def count_words(path):
#     """Count the approximate number of words in a file."""
#     try:
#       contents = path.read_text(encoding = 'utf-8')
#     except FileNotFoundError:
#       print(f"Sorry, the file {path.stem} does not exist.")
#     else:
#       # Count the approximate number of words in the file.
#       words = contents.split()
#       num_words = len(words)
#       print(f"The file {path.name} contains {num_words} words in the file.")


# # path = base_dir/ "txt files" / "alice.txt"
# # count_words(path)

# # It’s a good habit to keep comments up to date when you’re modifying a program.

# filenames = ['alice.txt', 'vedas.txt', 'frankenstein.txt', 'sherlock.txt']

# for file in filenames:
#    path = base_dir / "txt files" / file
#    count_words(path)


# # try-except block advantages:
#    # prevent our users from seeing a traceback, 
#    # we let the program continue analyzing the texts it’s able to find.
#    # If we don’t catch the FileNotFoundError that vedas.txt raises, the user would see a full traceback
#    # and the program would stop running after trying to analyze vedas.
#    # It would never analyze frankenstein or sherlock.


# '''FAILING SILENTLY'''

# # you’ll want the program to fail silently when an exception occurs and continue on as if nothing happened.
# # Python has a pass statement that tells it to do nothing in a block

# from pathlib import Path
# base_dir = Path(__file__).parent

# def count_words(path):
#    """Count the approximate number of words in a file."""
#    try:
#       contents = path.read_text(encoding = 'utf-8')
#    except FileNotFoundError:
#       pass
#    else:
#       "Count the approximate number of words in the file."
#       words = contents.split()
#       num_words = len(words)
#       print(f"The file {file} contains {num_words} words.")
#       print(f"The {path.stem} book contains {num_words} words.")

# filenames = ['alice.txt', 'vedas.txt', 'frankenstein.txt', 'sherlock.txt']

# for file in filenames:
#    path = base_dir/ "txt files" / file
#    count_words(path)


# # Now when a FileNotFoundError is raised, the code in the except block runs, but nothing happens.
# # No traceback is produced, and there’s no output in response to the error that was raised.
# # Users see the word counts for each file that exists, but they don’t see any indication that a file wasn’t found.

# '''
# The pass statement also acts as a placeholder. 
# It’s a reminder that you’re choosing to do nothing 
# at a specific point in your program’s execution and 
# that you might want to do something there later
# '''

# '''DECIDE WHICH ERRORS TO REPORT'''

# # Python’s error-handling structures 
# # give you fine-grained control over how much to share with users when things go wrong

# # Well-written, properly tested code is not very prone to internal errors, such as syntax or logical errors.
# # But every time your program depends on something external such as user input, the existence of a file,
# # or the availability of a network connection, there is a possibility of an exception being raised.


# '''TRY IT YOURSELF'''

# # 10.6.  ADDITION

# while True:
#    """Adding two numbers"""
     
#    try:
#       first_num = input("Enter the first number: ").lower()
#       if first_num == 'q':
#          break
      
#       second_num = input("Enter the second number: ").lower()
#       if second_num == 'q':
#          break

#       result = int(first_num) + int(second_num)
#       print(f"Total: {result}")
#    except ValueError:
#       print("Please enter a number only or enter 'q' to quit.")


# # 10.7.  ADDITION CALCULATOR

# while True:
#    """Adding two numbers"""
     
#    try:
#       first_num = input("Enter the first number: ").lower()
#       if first_num == 'q':
#          break
#       first_num = int(first_num)

#       second_num = input("Enter the second number: ").lower()
#       if second_num == 'q':
#          break
#       second_num = int(second_num)

#    except ValueError:
#       print("Please enter a number only or enter 'q' to quit.")
   
#    else: 
#       result = first_num + second_num
#       print(f"Total: {result}")
#       print("Enter first number to continue, or enter 'q' to quit.")


# # 10.8.  CATS and DOGS

# from pathlib import Path
# base_dir = Path(__file__).parent

# # path = base_dir/ "txt files"/ "cats.txt"
# # contents = path.read_text('utf-8')
# # lines = contents.splitlines()
# # for line in lines:
# #    print(f"{line.title()}")

# try:

#    path = base_dir/ "txt files"/ "cats.txt"
#    contents = path.read_text('utf-8')
#    lines = contents.splitlines()
#    for line in lines:
#       print(f"{line.title()}")

#    path1 = base_dir/ "dogs.txt"
#    # path1 = base_dir/ "txt files" /"dogs.txt"
#    contents1 = path1.read_text('utf-8')
#    lines1 = contents1.splitlines()
#    for line in lines1:
#       print(line.title())

# except FileNotFoundError:
#    print(f"Sorry! couldn't locate the {path1.stem} text file.")

# # 10.9.  Silent Cats and Dogs

# from pathlib import Path
# base_dir = Path(__file__).parent

# try:

#    path = base_dir/ "txt files"/ "cats.txt"
#    contents = path.read_text('utf-8')
#    lines = contents.splitlines()
#    for line in lines:
#       print(f"{line.title()}")

#    path1 = base_dir/ "dogs.txt"
#    # path1 = base_dir/ "txt files" /"dogs.txt"
#    contents1 = path1.read_text('utf-8')
#    lines1 = contents1.splitlines()
#    for line in lines1:
#       print(line.title())

# except FileNotFoundError:
#    pass


# 10.10. Common Words

from pathlib import Path

base_dir = Path(__file__).parent
path = base_dir/ "txt files"/ "frankenstein.txt"

words = path.read_text('utf-8')
num_words = len(words)
print(num_words)  # 438840

# Counting the words with 'the'
print(words.count("the"))  # 5475   (includes 'the', 'then', 'there', 'them' etc.)
print(words.count('the ')) # 3683   (includes 'the' only)

# First 25 words containing 'the'

words = path.read_text('utf-8').lower().split()
the_words = [word for word in words if 'the' in word]
print(the_words[:25])

# Words that start with "the"

start_with_the = [word for word in words if word.startswith("the")]
print(start_with_the[:25])

# Words that end with 'the'

end_with_the = [word for word in words if word.endswith("the")]
print(end_with_the[-25:])

'''Better approach using regex'''

from pathlib import Path
import re

text = path.read_text('utf-8').lower()
words = re.findall(r'\b\w+\b', text)

#1. Contains 'the'
print("\nContains 'the': ")
print([word for word in words if 'the' in word][:25])

#2. Starts with 'the'
print("\nStarts with 'the': ")
print([w for w in words if w.startswith("the")][:25])

#3. Ends with 'the'
print("\nEnds with 'the': ")
print([w for w in words if w.endswith('the')][-25:])