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


'''PLOTTING AND STYLING INDIVIDUAL POINTS WITH SCATTER()'''

# Sometimes, it’s useful to plot and style individual points based on certain characteristics.
# For example, you might plot small values in one color and larger values in a different color.
# You could also plot a large dataset with one set of styling options.

# To plot a single point, pass the single x- and y-values of the point to scatter()

import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.scatter(2, 4)
plt.show()

'STYPLE THE OUTPUT'