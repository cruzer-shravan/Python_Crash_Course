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

'''TRY IT YOURSELF'''

# 10.4. Guest

user_name = input("Enter your name: ").lower()

from pathlib import Path

base_dir = Path(__file__).parent
path = base_dir / "guest.txt"

user = ''
while True:
    print("Enter your name or type 'q' to exit!")
    if user_name == 'q':
        break
    else:
        user += user_name
        path.write_text(user)