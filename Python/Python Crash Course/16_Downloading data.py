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


# # 16.4. Automatic Indexes

# from pathlib import Path
# import csv
# from datetime import datetime
# import matplotlib.pyplot as plt

# path = Path('weather_data/death_valley_2021_simple.csv')
# lines = path.read_text().splitlines()

# reader = csv.reader(lines)
# # print(reader)
# header_row = next(reader)
# print(header_row)

# date_index = header_row.index('DATE')
# high_index = header_row.index('TMAX')
# low_index = header_row.index('TMIN')
# name_index = header_row.index('NAME')

# # for index, column_header in enumerate(header_row):
# #     print(index, column_header)

# # Extract dates, high and low temperatures.
# dates, highs, lows = [], [], []
# place_name = ""

# for row in reader:
#     # Grab the station name, if it's not already set.
#     if not place_name:
#         place_name = row[name_index]
    
#     current_date = datetime.strptime(row[date_index], '%Y-%m-%d')
#     try:
#         high = int(row[high_index])
#         low = int(row[low_index])
#     except ValueError:
#         print(f"Missing data for {current_date}")
#     else:
#         dates.append(current_date)
#         highs.append(high)
#         lows.append(low)

# max_temp = max(highs) 
# min_temp = min(lows) 
# margin = round((max_temp + min_temp) * 0.1 )
# print(max_temp, min_temp, margin)

# # Plot the high temperatures.
# plt.style.use('seaborn-v0_8-pastel')
# fig, ax = plt.subplots()
# ax.plot(dates, highs, color = 'red', alpha=0.5)     # alpha argument controls colour transparency (0- complete transparency, 1 - complete opaque)
# ax.plot(dates, lows, color = 'blue', alpha=0.5)     # facecolor determines the color of the shaded region between highs & lows.
# ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

# # Format plot.
# ax.set_title(f"Daily High and Low Temperatures, 2021\n{place_name}", fontsize = 20)
# ax.set_xlabel("", fontsize = 16)
# fig.autofmt_xdate()         # draws the date labels diagonally to prevent them from overlapping
# ax.set_ylabel("Temperature (F)", fontsize = 16)
# ax.tick_params(labelsize = 16)
# # ax.set_ylim(10, 140)
# ax.set_ylim(min_temp - margin, max_temp + margin)

# plt.show()

# '''MAPPING GLOBAL DATASETS: GeoJSON Format'''

# # Using Plotly’s scatter_geo() plot, you’ll create visualizations that clearly
# # show the global distribution of earthquakes. 
# # The data is stored in GeoJSON format and we will use JSON module to work with the data.

# '''Examining GeoJSON Data'''

# '''
# {"type":"FeatureCollection","metadata":{"generated":164905229
# 6000,...
# {"type":"Feature","properties":{"mag":1.6,"place":"63 km SE o
# f Ped...
# '''

# # This file is formatted more for machines than humans. 
# # But we can see that the file contains some dictionaries, as well
# # as information that we’re interested in, such as earthquake magnitudes and locations

# # The json module provides a variety of tools for exploring and working with JSON data.
# # Some of these tools will help us reformat the file so we can look at the raw data more easily
# # before we work with it programmatically.

# from pathlib import Path
# import json

# # Read data as string and convert to a Python object.
# path = Path('eq_data/eq_data_1_day_m1.geojson')
# contents = path.read_text(encoding= 'utf-8')
# all_eq_data = json.loads(contents)

# # Create a more readable version of the data file.
# path = Path('eq_data/readable_eq_data.geojson')
# readable_contents = json.dumps(all_eq_data, indent=4)
# path.write_text(readable_contents)

# # json.loads() to convert the string representation of the file to a Python object
# # json.dumps() function optional indent argument, which 
# # tells it how much to indent nested elements in the data structure

# # This GeoJSON file has a structure that’s helpful for location-based data.
# # This structure might look confusing, but it’s quite powerful.
# # It allows geologists to store as much information as they
# # need to in a dictionary about each earthquake, and then
# # stuff all those dictionaries into one big list

# # The key "properties" contains a lot of information about each earthquake.
# # We’re mainly interested in the magnitude of each earthquake, associated with the key "mag".

# # NOTE:
# # This convention probably arose because humans discovered latitude long before 
# # we developed the concept of longitude. However, many geospatial frameworks list
# # the longitude first and then the latitude, because this corresponds to the (x, y)
# # convention we use in mathematical representations. 
# # The GeoJSON format follows the (longitude, latitude) convention.

# '''Making a List of All Earthquakes'''

# # Examine all earthquakes in the dataset.
# all_eq_dicts = all_eq_data['features']
# # print(len(all_eq_data))     # 160

# '''Extracting Magnitudes'''

# mags = []
# for eq_dict in all_eq_dicts:
#     mag = eq_dict['properties']['mag']
#     mags.append(mag)

# print(mags[:10])

# '''Extracting Location Data'''

# # The location data for each earthquake is stored under the key "geometry". 
# # Inside the geometry dictionary is a "coordinates" key, and the first 
# # two values in this list are the longitude and latitude.

# mags, lons, lats = [], [], []
# for eq_dict in all_eq_dicts:
#     mag = eq_dict['properties']['mag']
#     lon = eq_dict['geometry']['coordinates'][0]
#     lat = eq_dict['geometry']['coordinates'][1]
#     mags.append(mag)
#     lons.append(lon)
#     lats.append(lat)

# print(mags[:10])
# print(lons[:10])
# print(lats[:10])

# '''Building a World Map - Simple map'''

# import plotly.express as px
# title = 'Global Earthquakes'
# fig = px.scatter_geo(lat=lats, lon=lons, title=title)
# fig.show()

# # The scatter_geo() function ❶ allows you to
# # overlay a scatterplot of geographic data on a map.

# '''Representing Magnitudes'''

# from pathlib import Path
# import json

# # Read data as string and convert to a Python object.
# path = Path('eq_data/eq_data_30_day_m1.geojson')
# contents = path.read_text(encoding= 'utf-8')
# all_eq_data = json.loads(contents)

# # Create a more readable version of the data file.
# path = Path('eq_data/readable_eq_data.geojson')
# readable_contents = json.dumps(all_eq_data, indent=4)
# path.write_text(readable_contents)

# '''Making a List of All Earthquakes'''

# # Examine all earthquakes in the dataset.
# all_eq_dicts = all_eq_data['features']
# # print(len(all_eq_data))     # 160

# '''Extracting Magnitudes'''

# mags = []
# for eq_dict in all_eq_dicts:
#     mag = eq_dict['properties']['mag']
#     mags.append(mag)

# print(mags[:10])

# mags, lons, lats = [], [], []
# for eq_dict in all_eq_dicts:
#     mag = eq_dict['properties']['mag']
#     lon = eq_dict['geometry']['coordinates'][0]
#     lat = eq_dict['geometry']['coordinates'][1]
#     mags.append(mag)
#     lons.append(lon)
#     lats.append(lat)

# print(mags[:10])
# print(lons[:10])
# print(lats[:10])

# '''Building a World Map - Simple map'''

# import plotly.express as px
# title = 'Global Earthquakes'
# # fig = px.scatter_geo(lat=lats, lon=lons, size=mags, title=title)
# # fig.show()


# '''Customizing Marker Colors'''

# # We can use Plotly’s color scales to customize each marker’s
# # color, according to the severity of the corresponding earthquake.

# fig = px.scatter_geo(lat=lats, lon=lons, size=mags, title=title,
#                      color=mags,                
#                      color_continuous_scale='Viridis',
#                      labels={'color': 'Magnitude'},
#                      projection='natural earth',
#                      )
# fig.show()

# We use the mags list to determine the color for each point, just as we did with the size argument.
# The color_continuous_scale argument tells Plotly which color scale to use.
# Viridis is a color scale that ranges from dark blue to bright yellow, and it works well for this dataset.
# We only need to set one custom label on this chart, making sure the color scale is labeled Magnitude instead of color.
# The projection argument accepts a number of common map projections. 
# Here we use the 'natural earth' projection, which rounds the ends of the map. 
# It’s common practice to add a trailing comma so you’re always ready to add another argument on the next line.

'''OTHER COLOR SCALES'''

# You can choose from a number of other color scales.
import plotly.express as px
# print(px.colors.named_colorscales())


'''ADDING HOVER TEXT'''
# To finish this map, we’ll add some informative text that appears when you hover over the marker representing an earthquake.

# mags, lons, lats, eq_titles = [], [], [], []
# for eq_dict in all_eq_dicts:
#     mag = eq_dict['properties']['mag']
#     lon = eq_dict['geometry']['coordinates'][0]
#     lat = eq_dict['geometry']['coordinates'][1]
#     eq_title = eq_dict['properties']['title']
#     mags.append(mag)
#     lons.append(lon)
#     lats.append(lat)
#     eq_titles.append(eq_title)

# print(mags[:10])
# print(lons[:10])
# print(lats[:10])

# import plotly.express as px
# title = 'Global Earthquakes'
# fig = px.scatter_geo(lat=lats, lon=lons, size=mags, title=title,
#                      color=mags,                
#                      color_continuous_scale='Viridis',
#                      labels={'color': 'Magnitude'},
#                      projection='natural earth',
#                      hover_name=eq_titles,
#                      )
# fig.show()


'''TRY IT YOURSELF'''

# 16.6. REFACTORING


# mags, lons, lats, eq_titles = [], [], [], []
# for eq_dict in all_eq_dicts:
#     mags.append(eq_dict['properties']['mag'])
#     lons.append(eq_dict['geometry']['coordinates'][0])
#     lats.append(eq_dict['geometry']['coordinates'][1])
#     eq_titles.append(eq_dict['properties']['title'])

# print(mags[:10])
# print(lons[:10])
# print(lats[:10])

# import plotly.express as px
# title = eq_dict['metadata']['title']
# fig = px.scatter_geo(lat=lats, lon=lons, size=mags, title=title,
#                      color=mags,                
#                      color_continuous_scale='Viridis',
#                      labels={'color': 'Magnitude'},
#                      projection='natural earth',
#                      hover_name=eq_titles,
#                      )
# fig.show()


# # 16.7. Automated Title

# from pathlib import Path
# import json

# path = Path('eq_data/eq_data_30_day_m1.geojson')
# contents = path.read_text(encoding= 'utf-8')
# all_eq_data = json.loads(contents)

# path = Path('eq_data/readable_eq_data.geojson')
# readable_contents = json.dumps(all_eq_data, indent=4)
# path.write_text(readable_contents)

# # Examine all earthquakes in the dataset.
# all_eq_dicts = all_eq_data['features']

# mags = []
# for eq_dict in all_eq_dicts:
#     mag = eq_dict['properties']['mag']
#     mags.append(mag)

# mags, lons, lats, eq_titles = [], [], [], []
# for eq_dict in all_eq_dicts:
#     mags.append(eq_dict['properties']['mag'])
#     lons.append(eq_dict['geometry']['coordinates'][0])
#     lats.append(eq_dict['geometry']['coordinates'][1])
#     eq_titles.append(eq_dict['properties']['title'])

# import plotly.express as px
# title = all_eq_data['metadata']['title']            # change
# fig = px.scatter_geo(lat=lats, lon=lons, size=mags, title=title,
#                      color=mags,                
#                      color_continuous_scale='Viridis',
#                      labels={'color': 'Magnitude'},
#                      projection='natural earth',
#                      hover_name=eq_titles,
#                      )
# fig.show()


# 16.8. World Fires

from pathlib import Path
import csv

path = Path('eq_data/world_fires_1_day.csv')
lines = path.read_text().splitlines()

reader = csv.reader(lines)
header_row = next(reader)

for index, column_header in enumerate(header_row):
    print(index, column_header)

'''Extracting and Reading Data'''

# Extract brightness
brightness, lats, lons  = [], [], []

for row in reader:
    try:
        lat = float(row[0])
        lon = float(row[1])
        bright = float(row[2])
    except ValueError:
        # Show raw date information for invalid rows.
        print(f"Invalid data for {row[5]}")
    else:
        lats.append(lat)
        lons.append(lon)
        brightness.append(bright)

# for row in reader:
#     brightness.append(round(float(row[2])))
#     lats.append(float(row[0]))
#     lons.append(float(row[1]))


import plotly.express as px
title = 'World fires'
fig = px.scatter_geo(lat=lats, lon=lons, size=brightness, title=title,
                     color=brightness,                
                     color_continuous_scale='Viridis',
                     labels={'color': 'brightness'},
                     projection='natural earth',
                     )
fig.show()

# import plotly.express as px
# print(px.colors.named_colorscales())