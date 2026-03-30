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


'''Handling the FileNotFoundError Exception'''

# One common issue when working with files is handling missing files
# (different location, incorrect filename or the file might not exist at all.)

from pathlib import Path
path = Path('alice.txt')    # file not saved in cwd

# contents = path.read_text(encoding='utf-8')

# we’re using read_text() in a slightly different way.
# encoding argument is needed when your system’s default encoding doesn’t match
# the encoding of the file that’s being read.
# reading from a file that wasn’t created on your system.

# Python can’t read from a missing file, so it raises an exception: 'FileNotFoundError'

# It’s often best to start at the very end of the complex traceback.
# This is important because it tells us what kind of exception to use in the except block that we’ll write.


try:
   contents = path.read_text(encoding="utf-8") 
except FileNotFoundError:
   print(f"Sorry, the file {path} does not exist.")

# the code in the try block produces a FileNotFoundError,
# so we write an except block that matches that error

