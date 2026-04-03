# '''GENERATING DATA'''

# # When a representation of a dataset is simple and visually appealing,
# # its meaning becomes clear to viewers. People will see patterns and
# # significance in your datasets that they never knew existed.

# # you can analyze non-numerical data as well.
# # People use Python for data-intensive work in genetics, climate research, 
# # political and economic analysis, and much more.
# # Data scientists have written an impressive array of visualization and analysis tools in Python

# 'MATPLOTLIB - a mathematical plotting library'

# # Used to make simple plots like line graphs and scatter plots.

# 'Concept of RANDOM WALK'
# # Visualisation generated from a series of random decisions.

# 'PLOTLY'

# # This package creates visualizations that work well on digital devices,
# # to analyze the results of rolling dice.
# # Plotly generates visualizations that automatically resize to fit a variety of display devices.
# # These visualizations can also include a number of interactive features

# 'Installing Matplotlib'

# try:
#     import matplotlib
#     print("Matplotlib version:", matplotlib.__version__)
# except ImportError:
#     print("Matplotlib is not installed")

# '''Plotting a simple Line Graph'''

# import matplotlib.pyplot as plt

# squares = [1, 4, 9, 16, 25]

# # fig, ax = plt.subplots()    # subplots - function, ax - variable (represents a single plot in the figure)
# # ax.plot(squares)            # plot - method
# # plt.show()                  # plt.show - function

# # The pyplot module contains a number of functions that help generate charts and plots.
# # plot() method, tries to plot the data it’s given in a meaningful way.
# # The function plt.show() opens Matplotlib’s viewer and displays the plot.


# '''Changing the Label Type and Line Thickness'''

# # Label type is too small and the line is a little thin to read easily.

# "Customise to improve the plot's readability"

# fig, ax = plt.subplots()
# ax.plot(squares, linewidth = 3)

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 14)
# ax.set_ylabel("Square of Value", fontsize = 14)

# # Set size of tick labels.
# ax.tick_params(labelsize = 14)  # size of values on both axes
# plt.show()

# '''CORRECTING THE PLOT - Data is not plotted correctly'''

# # When you give plot() a single sequence of numbers, 
# # it assumes the first data point corresponds to an x-value of 0,
# # but our first point corresponds to an x-value of 1.

# import matplotlib.pyplot as plt

# input_values = [1, 2, 3, 4, 5]
# squares = [1, 4, 9, 16, 25]

# fig, ax = plt.subplots()
# ax.plot(input_values, squares, linewidth = 3)

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 14)
# ax.set_ylabel("Square of Value", fontsize = 14)

# # Set size of tick labels.
# ax.tick_params(labelsize = 14)  # size of values on both axes
# plt.show()

'''USING BUILT-IN STYLES'''

# Matplotlib has a number of predefined styles available.
# They can make your visualizations appealing without requiring much customization.

# >>> import matplotlib.pyplot as plt
# >>> plt.style.available
['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 
 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 
 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 
 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 
 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 
 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

# To use any of these styles, add one line of code before calling subplots():


# import matplotlib.pyplot as plt

# input_values = [1, 2, 3, 4, 5]
# squares = [1, 4, 9, 16, 25]

# plt.style.use('tableau-colorblind10')
# fig, ax = plt.subplots()
# ax.plot(input_values, squares, linewidth = 3)

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 14)
# ax.set_ylabel("Square of Value", fontsize = 14)

# # Set size of tick labels.
# ax.tick_params(labelsize = 14)  # size of values on both axes
# plt.show()


# '''PLOTTING AND STYLING INDIVIDUAL POINTS WITH SCATTER()'''

# # Sometimes, it’s useful to plot and style individual points based on certain characteristics.
# # For example, you might plot small values in one color and larger values in a different color.
# # You could also plot a large dataset with one set of styling options.

# # To plot a single point, pass the single x- and y-values of the point to scatter()

# import matplotlib.pyplot as plt

# plt.style.use('dark_background')
# fig, ax = plt.subplots()
# # ax.scatter(2, 4)
# # plt.show()

# 'STYLE THE OUTPUT'

# ax.scatter(2, 4,s=200)

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 12)
# ax.set_ylabel("Square of Value", fontsize = 12)

# # Set size of tick labels.
# ax.tick_params(labelsize = 12)
# plt.show()

# '''Plotting a series of points with scatter()'''

# # To plot a series of points, we can pass scatter() separate lists of x- and y-values

# import matplotlib.pyplot as plt

# x_values = [1, 2, 3, 4, 5]
# y_values = [1, 4, 9, 16, 25]

# plt.style.use('dark_background')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, s=100)   # s=100 --> size of dot

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 12)
# ax.set_ylabel("Square of Value", fontsize = 12)

# # Set size of tick labels.
# ax.tick_params(labelsize = 12)
# plt.show()

# # Matplotlib reads one value from each list as it plots each point. 
# # The points to be plotted are (1, 1), (2, 4), (3, 9), (4, 16), and (5, 25)

# '''Calculating Data Automatically'''

# # Writing lists by hand can be inefficient, especially when we have many points.
# # Rather than writing out each value, let’s use a loop to do the calculations for us.

# import matplotlib.pyplot as plt

# x_values = range(1, 1001)
# y_values = [x**2 for x in x_values]

# plt.style.use('ggplot')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, s=10)   # s=100 --> size of dot

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 12)
# ax.set_ylabel("Square of Value", fontsize = 12)

# # Set size of tick labels.
# ax.tick_params(labelsize = 12)

# # Set the range for each axis
# ax.axis([0, 1100, 0, 1_100_000])
# plt.show()      # Python can plot 1,000 points as easily as it plots 5 points.


# '''Customizing Tick Labels'''

# # When the numbers on an axis get large enough, 
# # Matplotlib defaults to scientific notation for tick labels.
# # This is usually a good thing, because larger numbers in plain notation take
# # up a lot of unnecessary space on a visualization.
# # Almost every element of a chart is customizable,
# # so you can tell Matplotlib to keep using plain notation if you prefer.

# import matplotlib.pyplot as plt

# x_values = range(1, 1001)
# y_values = [x**2 for x in x_values]

# plt.style.use('ggplot')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, s=10)   # s=100 --> size of dot

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 12)
# ax.set_ylabel("Square of Value", fontsize = 12)

# # Set size of tick labels.
# ax.tick_params(labelsize = 12)

# # Set the range for each axis
# ax.axis([0, 1100, 0, 1_100_000])
# ax.ticklabel_format(style='plain')    
# plt.show()

# # The ticklabel_format() method allows you to override the default tick label style for any plot.

# '''Defining Custom Colors'''

# # To change the color of the points, pass the argument color to scatter()

# ax.scatter(x_values, y_values, color = 'red', s=10) 

# ax.scatter(x_values, y_values, color = (0, 0.8, 0) , s=10)

# # Values closer to 0 produce darker colors, and values closer to 1 produce lighter colors

# ''' Using a Colormap '''

# # A colormap is a sequence of colors in a gradient that moves from a starting to an ending color.
# # Using a colormap ensures that all points in the visualization vary smoothly and accurately
# # along a well-designed color scale.

# # The pyplot module includes a set of built-in colormaps.
# # how to assign a color to each point, based on its y-value?

# import matplotlib.pyplot as plt

# x_values = range(1, 1001)
# y_values = [x**2 for x in x_values]

# plt.style.use('ggplot')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Greens, s=10)   # A plot using the Greens colormap.

# # Set chart title and label axes.
# ax.set_title("Square Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 12)
# ax.set_ylabel("Square of Value", fontsize = 12)

# # Set size of tick labels.
# ax.tick_params(labelsize = 12)

# # Set the range for each axis
# ax.axis([0, 1100, 0, 1_100_000])
# ax.ticklabel_format(style='plain')    
# plt.show()


# '''Saving Your Plots Automatically'''

# # If you want to save the plot to a file instead of showing it in the Matplotlib viewer, 
# # you can use plt.savefig() instead of plt.show().

# plt.savefig('square_plot.png', bbox_inches = 'tight')

# # The first argument is a filename for the plot image,
# # which will be saved in the same directory as scatter_squares.py.
# # The second argument trims extra whitespace from the plot.
# # If you want the extra whitespace around the plot,
# # you can omit this argument. You can also call savefig() with a Path object,
# # and write the output file anywhere you want on your system.


# '''TRY IT YOURSELF'''

# # 15.1. CUBES

# import matplotlib.pyplot as plt

# x_values = [1, 2, 3, 4, 5]
# y_values = [x**3 for x in x_values]

# plt.style.use('dark_background')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, s=100)

# # Set chart title and label axes
# ax.set_title("Cube Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 14)
# ax.set_ylabel("Cube of values", fontsize = 14)

# # Set size of tick labels
# ax.tick_params(labelsize = 12)
# plt.show()

# import matplotlib.pyplot as plt

# x_values = [1, 2, 3, 4, 5]
# y_values = [x**3 for x in x_values]

# plt.style.use('dark_background')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, s=100)

# # Set chart title and label axes
# ax.set_title("Cube Numbers", fontsize = 24)
# ax.set_xlabel("Value", fontsize = 14)
# ax.set_ylabel("Cube of values", fontsize = 14)

# # Set size of tick labels
# ax.tick_params(labelsize = 12)
# plt.show()


# # 15.2  Colored cubes - Applying colormap and plotting first 5000 cubic numbers

# import matplotlib.pyplot as plt

# x_values = range(1,5001)
# y_values = [x**3 for x in x_values]

# plt.style.use('ggplot')
# fig, ax = plt.subplots()
# ax.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues, s=5)

# # Set chart title and label axes
# ax.set_title("Cubes of Numbers", fontsize = 24)
# ax.set_xlabel("Values", fontsize = 14)
# ax.set_ylabel("Cubes of values", fontsize = 14)

# # Set size of tick labels
# ax.tick_params(labelsize=10)

# # Set the range for each axis
# ax.axis([0, 5100, 0, 126_000_000_000])
# ax.ticklabel_format(style='plain')
# plt.show()


# '''RANDOM WALKS'''

# # A random walk is a path that’s determined by a series of simple decisions, 
# # each of which is left entirely to chance.

# # Random walks have practical applications in nature, physics, biology, chemistry, and economics.
# # For example, a pollen grain floating on a drop of water.

# '''Creating the RandomWalk Class'''

# # This class needs 3 attributes.
# # one variable - to track the number of points in the walk.
# # two lists - to store the x and y corrdinates of each point in the walk.

# # We’ll only need two methods for the RandomWalk class: 
# # the __init__() method and fill_walk(), which will calculate the points in the walk

# '''Refer the 15.2_Random_walk.py file for Randomwalk class, its init method and fillwalk method'''

# '''PLOTTING THE RANDOM WALK'''

# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Make a random walk
# rw = Randomwalk()       # Random walk is created
# rw.fill_walk()          # called fill_walk through rw

# # Plot the points in the walk.
# plt.style.use('classic')
# fig, ax = plt.subplots()
# ax.scatter(rw.x_values, rw.y_values, s=15)
# ax.set_aspect('equal')
# plt.show()

# # rw.x_values, rw.y_values feed values to x, y axes to scatter to visualise the walk.
# # Matplotlib scales each axis independently.
# # aspect'equal' means both axes should have equal spacing btn tick marks.

# '''GENERATING MULTIPLE RANDOM WALKS'''

# # One way to use the preceding code to make multiple walks without having to
# # run the program several times is to wrap it in a while loop

# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk()       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots()
#     ax.scatter(rw.x_values, rw.y_values, s=15)
#     ax.set_aspect('equal')
#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break

# # If you generate a few walks, you should see some that stay near the starting point, 
# # some that wander off mostly in one direction, 
# # some that have thin sections connecting larger groups of points, 
# # and many other kinds of walks.


# '''STYLING THE WALK'''

# 'Customise our plots'

# # EMPHASIZE the important characteristics
# # Where the walk began, where it ended and the path taken.

# # DEEMPHASIZE distracting elements
# # such as tick marks and labels

# # The result should be a simple visual representation
# # that clearly communicates the path taken in each random walk.

# '''COLORING THE POINTS'''

# # Use a colormap
# # Remove the black outline from each dot, so that the color of the dots will be clearer.

# # To color the points according to their position in the walk,
# # we pass the c argument a list containing the position of each point.


# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk()       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots()
#     point_numbers = range(rw.num_points)            # new line
#     ax.scatter(rw.x_values, rw.y_values, c=point_numbers, 
#                cmap=plt.cm.Greens, edgecolors='none', s=15) # new line
#     ax.set_aspect('equal')
#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break


# '''PLOTTING THE STARTING AND ENDING POINTS'''

# # We’ll make the end points larger and color them differently to make them stand out.


# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk()       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots()
#     point_numbers = range(rw.num_points)            
#     ax.scatter(rw.x_values, rw.y_values, c=point_numbers, 
#                cmap=plt.cm.Greens, edgecolors='none', s=15)
#     ax.set_aspect('equal')
    
#     # Emphasize the first and last points.            # New block of code
#     ax.scatter(0, 0, c='blue', edgecolors='none', s=100)
#     ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none', s= 100)

#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break

# '''CLEANING UP THE AXES'''

# # Let’s remove the axes in this plot so they don’t distract from the path of each walk.

# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk()       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots()
#     point_numbers = range(rw.num_points)            # new line
#     ax.scatter(rw.x_values, rw.y_values, c=point_numbers, 
#                cmap=plt.cm.Greens, edgecolors='none', s=15) # new line
#     ax.set_aspect('equal')
    
#     # Emphasize the first and last points.
#     ax.scatter(0, 0, c='blue', edgecolors='none', s=100)
#     ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none', s= 100)

#     # Remove the axes.  (series of plots with no axes)      # New block of code
#     ax.get_xaxis().set_visible(False)   
#     ax.get_yaxis().set_visible(False)

#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break

# '''ADDING PLOT POINTS'''

# # Let’s increase the number of points, to give us more data to work with.
# # To do so, we increase the value of num_points, when we make a RandomWalk instance
# # and adjust the size of each dot when drawing the plot.


# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk(500_000)       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots()
#     point_numbers = range(rw.num_points)
#     ax.scatter(rw.x_values, rw.y_values, c=point_numbers, 
#                cmap=plt.cm.Reds, edgecolors='none', s=2) 
#     ax.set_aspect('equal')
    
#     # Emphasize the first and last points.
#     ax.scatter(0, 0, c='blue', edgecolors='none', s=100)
#     ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none', s= 100)

#     # Remove the axes.  (series of plots with no axes)      
#     ax.get_xaxis().set_visible(False)   
#     ax.get_yaxis().set_visible(False)

#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break


# '''ALTERING THE SIZE TO FILL THE SCREEN'''

# # A visualization is much more effective at communicating patterns in data if it fits nicely on the screen.


# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk(50_000)       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots(figsize = (15,9), dpi=128)       # New line
#     point_numbers = range(rw.num_points)            
#     ax.scatter(rw.x_values, rw.y_values, c=point_numbers, 
#                cmap=plt.cm.Reds, edgecolors='none', s=2) 
#     ax.set_aspect('equal')
    
#     # Emphasize the first and last points.
#     ax.scatter(0, 0, c='blue', edgecolors='none', s=100)
#     ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none', s= 100)

#     # Remove the axes.  (series of plots with no axes)     
#     ax.get_xaxis().set_visible(False)   
#     ax.get_yaxis().set_visible(False)

#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break


# '''TRY IT YOURSELF'''

# # 15.3. MOLECULAR MOTION


# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk(5_000)       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots(figsize = (15,9), dpi=128)       # New line
#     point_numbers = range(rw.num_points)            
#     ax.plot(rw.x_values, rw.y_values, c='red', linewidth = 2) 
#     ax.set_aspect('equal')
    
#     # # Emphasize the first and last points.
#     ax.scatter(0, 0, c='blue', edgecolors='none', s=100)
#     ax.scatter(rw.x_values[-1], rw.y_values[-1], c='green', edgecolors='none', s= 100)

#     # Remove the axes.  (series of plots with no axes)     
#     ax.get_xaxis().set_visible(False)   
#     ax.get_yaxis().set_visible(False)

#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break


# # 15.4. Modified Random Walks


# import matplotlib.pyplot as plt
# from random_walk import Randomwalk

# # Keep making new walks, as long as the program is active.
# while True:
#     # Make a random walk
#     rw = Randomwalk(50_000)       # Random walk is created
#     rw.fill_walk()          # called fill_walk through rw

#     # Plot the points in the walk.
#     plt.style.use('classic')
#     fig, ax = plt.subplots(figsize = (15,9), dpi=128)       # New line
#     point_numbers = range(rw.num_points)            
#     ax.scatter(rw.x_values, rw.y_values, c=point_numbers, 
#                cmap=plt.cm.Reds, edgecolors='none', s=2) 
#     ax.set_aspect('equal')
    
#     # Emphasize the first and last points.
#     ax.scatter(0, 0, c='blue', edgecolors='none', s=100)
#     ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none', s= 100)

#     # Remove the axes.  (series of plots with no axes)     
#     ax.get_xaxis().set_visible(False)   
#     ax.get_yaxis().set_visible(False)

#     plt.show()

#     keep_running = input("Make another walk? (y/n): ")
#     if keep_running == 'n':
#         break

# # Output 1:  x-direction = 1, y_direction = 1, -1
#     # There was a thin horizontal line that moved along the x axis.

# # Output 2: x_direction = 1, y_direction = 1
#     # A diagonal line was formed between the axes, in Quadrant 1
#     # showing an increasing trend with every increase in x and y coordinates.

# # Output 3: x direction = -1 and y direction = 1
#     # A diagonal line formed between the axes in Quadrant 2
#     # showing an increasing trend with every decrease in x values and increase in y values.

# # Output 4: x = -1, y = -1
#     # Decreasing trend diagonally towards the origin


# # 15.5. Refactoring

# # Replaced x and y coordinate steps to just one step function in random_walk Class.


'''ROLLING DICE WITH PLOTLY'''

# Plotly produce interactive visualizations that will be displayed in a browser.
# Visualizations will scale automatically to fit the viewer's screen.

# PLOTLY EXPRESS, a subset of Plotly that focuses on generating
# plots with as little code as possible

# Eg. Analyze the results of rolling dice.
# We'll try to determine which numbers are most likely to occur by generating a dataset
# that represents rolling dice.

# It also relates to many real-world situations where randomness plays a significant factor.


'''INSTALLING PLOTLY'''

# Plotly Express depends on pandas, which is a library for working efficiently with data.

'Creating the Die Class'

# from random import randint        # Pushed it to another die.py file

# class Die:
#     """A class representing a single die"""

#     def __init__(self, num_sides = 6):
#         """Assume a six sided die"""
#         self.num_sides = num_sides
    
#     def roll(self):     # roll() method
#         """Return a random value between 1 and number of sides."""
#         return randint(1, self.num_sides)       # randint() function
    

# '''Rolling the die'''

# from die import Die

# # Create a D6
# die = Die()

# # Make some rolls and store results in a list.
# results = []
# for roll_num in range(100):
#     result = die.roll()
#     results.append(result)
# print(results)

'''Analyzing the Results'''

from die import Die

die = Die()

results = []
for roll_num in range(101):
    result = die.roll()
    results.append(result)

# Analyzing the results

frequencies =[]
poss_results = range(1, die.num_sides + 1)
for value in poss_results:
    frequency = results.count(value)
    frequencies.append(frequency)

print(frequencies)

# Making a histogram

import plotly.express as px
