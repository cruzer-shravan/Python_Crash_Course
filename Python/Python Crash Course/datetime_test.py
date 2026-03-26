"""datetime Module"""

# To convert it into HH:MM:SS + microseconds, we use the datetime module.

'''Correct way to Format time'''

from datetime import datetime

start_time = datetime.now()                 

print("Start time:", start_time.strftime("%H:%M:%S.%f"))        # OUTPUT: Start time: 13:16:17.346616

'''Renamed the file name from datetime.py to datetime_test.py due to filename "shadow built-in modules (datetime)"'''

# 🚀 Pro Tip for You (as future ML engineer)

# This issue comes up a LOT in real projects.

# ✔ Quick Debug Trick:

import datetime
print(datetime.__file__)        # OUTPUT: C:\Users\cruze\AppData\Local\Programs\Python\Python310\lib\datetime.py

# 🔍 Why this confirms the fix

    # Earlier, the error showed:

    # ImportError: cannot import name 'datetime' from partially initialized module 'datetime' (most likely due to a circular import) 
    # (d:\Data science\Programming Languages\Python\Elite\Python\Python Crash Course\datetime.py)       # ACCESSING MODULE (FILE NAME)

# ❌ That meant your file was shadowing the module

    # Now it shows:

    # C:\Users\cruze\AppData\Local\Programs\Python\Python310\lib\datetime.py        # ACCESSING PROGRAM FILES


"""🟢 Apply to Your Code"""

# Replace your timing section with this:

from datetime import datetime
# from random import choice

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%H:%M:%S.%f')}")  # OUTPUT: Start time: 13:16:17.346616

# ---- your logic ---- #

end_time = datetime.now()
print(f"End time: {end_time.strftime('%H:%M:%S.%f')}")      # OUTPUT: End time: 13:16:17.346616

runtime = end_time - start_time
print(f"Runtime: {runtime}")                                # OUTPUT: Runtime: 0:00:00


"""🚀 Bonus: If you want higher precision (recommended for benchmarking)"""

# 👉 Use time.perf_counter(): This is more accurate for performance measurement than datetime.

import time

start = time.perf_counter()

# your code

end = time.perf_counter()

print(f"Runtime: {end - start:.6f} seconds")                # OUTPUT: Runtime: 0.000000 seconds

