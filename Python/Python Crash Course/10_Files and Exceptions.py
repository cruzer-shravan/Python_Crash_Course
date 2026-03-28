'''Reading from a file'''

'''Reading the contents of a file'''

from pathlib import Path

path = Path('D:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\pi_digits.txt')
contents = path.read_text()
print(contents)

contents = contents.rstrip()
print(contents)

contents = path.read_text().rstrip()
print(contents)


'''RELATIVE AND ABSOLUTE FILE PATHS'''

# To get Python to # open files from a directory other than the one where your program file is stored, 
# You need to provide the correct path.
# There are two main ways to specify paths in programming.

# path = Path('text_files/filename.txt')

# You can use an absolute path if a relative path doesn’t work.
# You’ll need to write out an absolute path to clarify where you want Python to look.

# path = Path('D:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\pi_digits.txt')

'''ACCESSING A FILE'S LINES'''

'''
For example, you might
want to read through a file of weather data and work with
any line that includes the word sunny in the description of
that day’s weather. In a news report, you might look for any
line with the tag <headline> and rewrite that line with a
specific kind of formatting.

You can use the splitlines() method to turn a long string
into a set of lines, and then use a for loop to examine each
line from a file, one at a time:
'''

# from pathlib import Path
# path = Path('pi_digits.txt')
contents = path.read_text()
lines = contents.splitlines()
for line in lines:
    print(line)
