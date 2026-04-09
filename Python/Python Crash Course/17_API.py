# program will use an application programming interface (API) to automatically
# request specific information from a website and then use that information to 
# generate a visualization. Because programs written like this will always use
# current data to generate a visualization, even when that data might be rapidly
# changing, the visualization will always be up to date.

'''Using and API'''

# An API is a part of a website designed to interact with programs.
# Those programs use very specific URLs to request certain information.
# This kind of request is called an API call. The requested data will be
# returned in an easily processed format, such as JSON or CSV. Most apps 
# that use external data sources, such as apps that integrate with 
# social media sites, rely on API calls.

'''Git and GitHub'''

# GitHub - a site that allows programmers to collaborate on coding projects.
# We’ll use GitHub’s API to request information about Python projects on the site, 
# and then generate an interactive visualization of the relative popularity of
# these projects using Plotly. 

# GitHub takes its name from Git, a distributed version control system.
# Git helps people manage their work on a project in a way that prevents changes
# made by one person from interfering with changes other people are making.

# Projects on GitHub are stored in repositories, which contain everything
# associated with the project: its code, information on its collaborators,
# any issues or bug reports, and so on.

# When users on GitHub like a project, they can “star” it to show their support 
# and keep track of projects they might want to use. 

# In this chapter, we’ll write a program to automatically download information 
# about the most-starred Python projects on GitHub, and then we’ll create an
# informative visualization of these projects.

'''Requesting Data using an API Call'''

# GitHub’s API lets you request a wide range of information through API calls.

'How an API call looks like ?'
'https://api.github.com/search/repositories?q=language:python+sort:stars'

# This call returns the number of Python projects currently hosted on GitHub,
# as well as information about the most popular Python repositories.

'Examine the call'

# The first part, https://api.github.com/, directs the request to the part
# of GitHub that responds to API calls. 

# The next part, search/repositories, tells the API to conduct a search through
# all the repositories on GitHub.

# The question mark after repositories signals that we’re about to pass an argument.

# The q stands for query, and the equal sign (=) lets us begin specifying a query (q=).

# By using language:python, we indicate that we want information only
# on repositories that have Python as the primary language.

# The final part, +sort:stars, sorts the projects by the number
# of stars they’ve been given.

'''
{
❶ "total_count": 8961993,
❷ "incomplete_results": true,
❸ "items": [
{
"id": 54346799,
"node_id": "MDEwOlJlcG9zaXRvcnk1NDM0Njc5OQ==",
"name": "public-apis",
"full_name": "public-apis/public-apis",
--snip--
'''

# The value for "incomplete_results" is true, which tells us that
# GitHub didn’t fully process the query ❷. 

# GitHub limits how long each query can run, in order to keep the API
# responsive for all users.

# The "items" returned are displayed in the list that follows, which
# contains details about the most popular Python projects on GitHub ❸