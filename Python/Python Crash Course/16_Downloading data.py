# # The ability to analyze data allows you to discover patterns and
# # connections that no one else has found.

# # Two common formats: CSV and JSON

# '''We’ll use Python’s csv module,
# to process weather data stored in the CSV format and
# analyze high and low temperatures over time in two different locations.'''

# '''The CSV File Format'''

# # One simple way to store data in a text file is to write the
# # data as a series of values separated by commas, called
# # comma-separated values. The resulting files are CSV files.

# # CSV files can be tedious for humans to read,
# # but programs can process and extract information from them quickly and accurately.

# '''Parsing the CSV File Headers'''

# # Python’s csv module in the standard library parses the lines in a CSV file
# # and allows us to quickly extract the values we’re interested in.

# from pathlib import Path
# import csv

# path = Path('weather_data/sitka_weather_07-2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)
# print(header_row)

# # splitlines() method --> get a list of all lines in the file, which we assign to lines.
# # Next, we build a reader object which is an object that can be used to parse each line in the file.
# # To make a reader object, call the function csv.reader() and pass it the list of lines from the CSV file.

# # When given a reader object, the next() function returns the
# # next line in the file, starting from the beginning of the file.

# ['STATION', 'NAME', 'DATE', 'TAVG', 'TMAX', 'TMIN']
# # The reader object processes the first line of commaseparated values in the file
# # and stores each value as an item in a list.

'''Printing the Headers and Their Positions'''

from pathlib import Path
import csv

# path = Path('weather_data/sitka_weather_07-2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# The enumerate() function returns both the index of each item
# and the value of each item as you loop through a list.

'''Extracting and Reading Data'''

# Extract high temperatures
# highs = []
# for row in reader:
#     high = int(row[4])
#     highs.append(high)
# print(highs)

'''Plotting Data in a Temperature Chart'''

# Simple plot of the daily highs using Matplotlib

import matplotlib.pyplot as plt

# Plot the high temperatures.

# plt.style.use('seaborn-v0_8-deep')
# fig, ax = plt.subplots()        # Equivalent to: fig = plt.figure()         ax = fig.add_subplot(111)
# ax.plot(highs, color='red')

# # Format plot.
# ax.set_title("Daily High Temperatures, July 2021", fontsize = 24)
# ax.set_xlabel("", fontsize = 16)
# ax.set_ylabel("Temperatur (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()

# # The datetime Module

# # convert the string "2021-07-01" (representing this date) to an object
# # by using the strptime() method from the datetime module. 

from datetime import datetime

# first_date = datetime.strptime('2021-07-01', '%Y-%m-%d')
# print(first_date)

# '''
# Argument            Meaning
#     %A      Weekday name, such as Monday
#     %B      Month name, such as January
#     %m      Month, as a number (01 to 12)
#     %d      Day of the month, as a number (01 to 31)
#     %Y      Four-digit year, such as 2019
#     %y      Two-digit year, such as 19
#     %H      Hour, in 24-hour format (00 to 23)
#     %I      Hour, in 12-hour format (01 to 12)
#     %p      AM or PM
#     %M      Minutes (00 to 59)
#     %S      Seconds (00 to 61) 
# '''

# '''Plotting Dates'''

# # Extract dates and high temperatures.
# dates, highs = [], []

# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     high = int(row[4])
#     dates.append(current_date)
#     highs.append(high)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red')

# # Format plot.
# ax.set_title("Daily High Temperatures, July 2021", fontsize = 24)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()


# 'Plotting a Longer Timeframe'

# # Let's use full year's worth of weather data for Sitka

# path = Path('weather_data/sitka_weather_2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates and high temperatures.
# dates, highs = [], []

# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     high = int(row[4])
#     dates.append(current_date)
#     highs.append(high)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red')

# # Format plot.
# ax.set_title("Daily High Temperatures, 2021", fontsize = 24)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()


'''Plotting a Second Data Series'''

# path = Path('weather_data/sitka_weather_2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates, high and low temperatures.
# dates, highs, lows = [], [], []

# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     high = int(row[4])
#     low = int(row[5])
#     dates.append(current_date)
#     highs.append(high)
#     lows.append(low)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red')
# ax.plot(dates, lows, color = 'blue')
# # Format plot.
# ax.set_title("Daily High and Low Temperatures, 2021", fontsize = 24)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()


'''Shading an Area in the Chart'''

# # Having added two data series, we can now examine the range of temperatures for each day. 
# # Let’s add a finishing touch to the graph by using shading to show the range 
# # between each day’s high and low temperatures. To do so, we’ll use the fill_between() method, 
# # which takes a series of x-values and two series of y-values and fills the space
# # between the two series of y-values

# path = Path('weather_data/sitka_weather_2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates, high and low temperatures.
# dates, highs, lows = [], [], []

# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     high = int(row[4])
#     low = int(row[5])
#     dates.append(current_date)
#     highs.append(high)
#     lows.append(low)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red', alpha=0.5)     # alpha argument controls colour transparency (0- complete transparency, 1 - complete opaque)
# ax.plot(dates, lows, color = 'blue', alpha=0.5)     # facecolor determines the color of the shaded region between highs & lows.
# ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)
# # Format plot.
# ax.set_title("Daily High and Low Temperatures, 2021", fontsize = 24)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()


'''Error Checking'''

# # Missing data can result in exceptions that crash our programs, unless we handle them properly.

# path = Path('weather_data/death_valley_2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates, high and low temperatures.
# dates, highs, lows = [], [], []

# '''for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     high = int(row[3])      # Here we face a ValueError: invalid literal for int() with base 10: ''
#     low = int(row[4])
#     dates.append(current_date)
#     highs.append(high)
#     lows.append(low)'''        

# # Value error - it can’t turn an empty string ('') into an integer
# # run error-checking code when the values are being read
# # from the CSV file to handle exceptions that might arise
# # Because the error is handled appropriately, our code is
# # able to generate a plot, which skips over the missing data.
# # Many datasets you work with will have missing,
# # improperly formatted, or incorrect data.
# # Here we used a try-except-else block to handle missing data.
# # Sometimes you’ll use continue to skip over some data,
# # or use remove() or del to eliminate some data after it’s been extracted


# 'Alternative for above code'
# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     try:
#         high = int(row[3])
#         low = int(row[4])
#     except ValueError:
#         print(f"Missing data for {current_date}")
#     else:
#         dates.append(current_date)
#         highs.append(high)
#         lows.append(low)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red', alpha=0.5)     # alpha argument controls colour transparency (0- complete transparency, 1 - complete opaque)
# ax.plot(dates, lows, color = 'blue', alpha=0.5)     # facecolor determines the color of the shaded region between highs & lows.
# ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

# # Format plot.
# ax.set_title("Daily High and Low Temperatures, 2021\nDeath Valley, CA", fontsize = 20)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()



# '''TRY IT YOURSELF'''

# # 16.1. Sitka Rainfall
# from pathlib import Path
# import csv
# from datetime import datetime

# path = Path('weather_data/sitka_weather_2021_full.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates, high and low temperatures.
# rainfall = []

# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     rain = float(row[5])
#     rainfall.append(rain)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-poster')
# fig, ax = plt.subplots()
# ax.plot(rainfall, color = 'blue')

# # # Format plot.
# ax.set_title("Daily rainfall 2021", fontsize = 20)
# ax.set_xlabel("No. of day in an year", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Measure of rainfall (mm)", fontsize = 16)
# ax.tick_params(labelsize = 16)

# plt.show()


# # 16.2. Sitka - Death Valley Comparison

# from pathlib import Path
# import csv
# from datetime import datetime
# import matplotlib.pyplot as plt

# path = Path('weather_data/death_valley_2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates, high and low temperatures.
# dates, highs, lows = [], [], []

# for row in reader:
#     current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     try:
#         high = int(row[3])
#         low = int(row[4])
#     except ValueError:
#         print(f"Missing data for {current_date}")
#     else:
#         dates.append(current_date)
#         highs.append(high)
#         lows.append(low)

# max_temp = max(highs) 
# min_temp = min(lows) 
# margin = round((max + min) * 0.1 )
# print(max_temp, min_temp, margin)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red', alpha=0.5)     # alpha argument controls colour transparency (0- complete transparency, 1 - complete opaque)
# ax.plot(dates, lows, color = 'blue', alpha=0.5)     # facecolor determines the color of the shaded region between highs & lows.
# ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

# # Format plot.
# ax.set_title("Daily High and Low Temperatures, 2021\nDeath Valley, CA", fontsize = 20)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)
# # ax.set_ylim(10, 140)
# ax.set_ylim(min_temp - margin, max_temp + margin)

# plt.show()


# # 16.3. San Francisco

# from pathlib import Path
# import csv
# from datetime import datetime
# import matplotlib.pyplot as plt

# path = Path('weather_data/san_francisco_weather_data.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# header_row = next(reader)
# print(header_row)

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# # Extract dates, high and low temperatures.
# highs, lows = [], []

# for row in reader:
#     # current_date = datetime.strptime(row[2], '%Y-%m-%d')
#     try:
#         high = float(row[2])
#         low = float(row[1])
#     except ValueError:
#         continue
#         # print(f"Missing data in row: {row}")
#     else:
#         # dates.append(current_date)
#         highs.append(high)
#         lows.append(low)

# highs_365 = highs[-365:]
# lows_365 = lows[-365:]

# max_temp = max(highs_365) 
# min_temp = min(lows_365) 
# margin = round((max_temp + min_temp) * 0.1 )
# print(max_temp, min_temp, margin)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(highs_365, color = 'red', alpha=0.5)     # alpha argument controls colour transparency (0- complete transparency, 1 - complete opaque)
# ax.plot(lows_365, color = 'blue', alpha=0.5)     # facecolor determines the color of the shaded region between highs & lows.
# ax.fill_between(range(len(highs_365)), highs_365, lows_365, facecolor='blue', alpha=0.1)    # fill_between requires 3 arguments, x, y1, y2

# # Format plot.
# ax.set_title("Daily High and Low Temperatures of San Francisco 2023", fontsize = 20)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (C)", fontsize = 16)
# ax.tick_params(labelsize = 16)
# # ax.set_ylim(10, 140)
# ax.set_ylim(min_temp - margin, max_temp + margin)

# plt.show()


# 16.4. Automatic Indexes

from pathlib import Path
import csv
from datetime import datetime
import matplotlib.pyplot as plt

path = Path('weather_data/death_valley_2021_simple.csv')
lines = path.read_text().splitlines()

reader = csv.reader(lines)
# print(reader)
header_row = next(reader)
print(header_row)

date_index = header_row.index('DATE')
high_index = header_row.index('TMAX')
low_index = header_row.index('TMIN')
name_index = header_row.index('NAME')

# for index, column_header in enumerate(header_row):
#     print(index, column_header)

# Extract dates, high and low temperatures.
dates, highs, lows = [], [], []
place_name = ""

for row in reader:
    # Grab the station name, if it's not already set.
    if not place_name:
        place_name = row[name_index]
    
    current_date = datetime.strptime(row[date_index], '%Y-%m-%d')
    try:
        high = int(row[high_index])
        low = int(row[low_index])
    except ValueError:
        print(f"Missing data for {current_date}")
    else:
        dates.append(current_date)
        highs.append(high)
        lows.append(low)

max_temp = max(highs) 
min_temp = min(lows) 
margin = round((max_temp + min_temp) * 0.1 )
print(max_temp, min_temp, margin)

# Plot the high temperatures.
plt.style.use('seaborn-v0_8-pastel')
fig, ax = plt.subplots()
ax.plot(dates, highs, color = 'red', alpha=0.5)     # alpha argument controls colour transparency (0- complete transparency, 1 - complete opaque)
ax.plot(dates, lows, color = 'blue', alpha=0.5)     # facecolor determines the color of the shaded region between highs & lows.
ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

# Format plot.
ax.set_title(f"Daily High and Low Temperatures, 2021\n{place_name}", fontsize = 20)
ax.set_xlabel("", fontsize = 16)
fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
ax.set_ylabel("Temperature (F)", fontsize = 16)
ax.tick_params(labelsize = 16)
# ax.set_ylim(10, 140)
ax.set_ylim(min_temp - margin, max_temp + margin)

plt.show()