# # program will use an application programming interface (API) to automatically
# # request specific information from a website and then use that information to 
# # generate a visualization. Because programs written like this will always use
# # current data to generate a visualization, even when that data might be rapidly
# # changing, the visualization will always be up to date.

# '''Using and API'''

# # An API is a part of a website designed to interact with programs.
# # Those programs use very specific URLs to request certain information.
# # This kind of request is called an API call. The requested data will be
# # returned in an easily processed format, such as JSON or CSV. Most apps 
# # that use external data sources, such as apps that integrate with 
# # social media sites, rely on API calls.

# '''Git and GitHub'''

# # GitHub - a site that allows programmers to collaborate on coding projects.
# # We’ll use GitHub’s API to request information about Python projects on the site, 
# # and then generate an interactive visualization of the relative popularity of
# # these projects using Plotly. 

# # GitHub takes its name from Git, a distributed version control system.
# # Git helps people manage their work on a project in a way that prevents changes
# # made by one person from interfering with changes other people are making.

# # Projects on GitHub are stored in repositories, which contain everything
# # associated with the project: its code, information on its collaborators,
# # any issues or bug reports, and so on.

# # When users on GitHub like a project, they can “star” it to show their support 
# # and keep track of projects they might want to use. 

# # In this chapter, we’ll write a program to automatically download information 
# # about the most-starred Python projects on GitHub, and then we’ll create an
# # informative visualization of these projects.

# '''Requesting Data using an API Call'''

# # GitHub’s API lets you request a wide range of information through API calls.

# 'How an API call looks like ?'
# 'https://api.github.com/search/repositories?q=language:python+sort:stars'

# # This call returns the number of Python projects currently hosted on GitHub,
# # as well as information about the most popular Python repositories.

# 'Examine the call'

# # The first part, https://api.github.com/, directs the request to the part
# # of GitHub that responds to API calls. 

# # The next part, search/repositories, tells the API to conduct a search through
# # all the repositories on GitHub.

# # The question mark after repositories signals that we’re about to pass an argument.

# # The q stands for query, and the equal sign (=) lets us begin specifying a query (q=).

# # By using language:python, we indicate that we want information only
# # on repositories that have Python as the primary language.

# # The final part, +sort:stars, sorts the projects by the number
# # of stars they’ve been given.

# '''
# {
# ❶ "total_count": 8961993,
# ❷ "incomplete_results": true,
# ❸ "items": [
# {
# "id": 54346799,
# "node_id": "MDEwOlJlcG9zaXRvcnk1NDM0Njc5OQ==",
# "name": "public-apis",
# "full_name": "public-apis/public-apis",
# --snip--
# '''

# # The value for "incomplete_results" is true, which tells us that
# # GitHub didn’t fully process the query ❷. 

# # GitHub limits how long each query can run, in order to keep the API
# # responsive for all users.

# # The "items" returned are displayed in the list that follows, which
# # contains details about the most popular Python projects on GitHub ❸


# '''Installing Requests'''

# # Requests package allows a Python program to easily
# # request information from a website and examine the response.

# 'python3 -m pip install --user requests'

# '''Processing an API response'''

# # We’ll write a program to automatically issue an API call and process the results
# import requests

# # Make an API call and check the response.
# url = "https://api.github.com/search/repositories"      # main part of url
# url += "?q=language:python+sort:stars+stars:>10000"     # query string with one more condition (stars:>10000)

# headers = {"Accept": "application/vnd.github.v3+json"}  # headers - response object
# r = requests.get(url, headers=headers)      # use request to make the call to the API.
# print(f"Status code: {r.status_code}")      # response object has an attribute called status_code, which tells whether the request was successful (200)

# # Convert the response object to a dictionary.
# response_dict = r.json()         

# # We asked the API to return the information in JSON format,
# # so we use the json() method to convert the information to a Python dictionary ❺. 
# # We assign the resulting dictionary to response_dict Process results.

# print(response_dict.keys())


# '''Working with the Response Dictionary'''

# # With the information from the API call represented as a
# # dictionary, we can work with the data stored there. 
# # Let’s generate some output that summarizes the information.

# print(f"Total repositories: {response_dict['total_count']}")
# print(f"Complete results: {not response_dict['incomplete_results']}")

# # Explore information about the repositories.
# repo_dicts = response_dict['items']     
# # The value associated with 'items' is a list containing a number of dictionaries, each of which contains data about an individual Python repository.

# print(f"Repositories returned: {len(repo_dicts)}")

# # Examine the first repository.
# repo_dict = repo_dicts[0]

# print(f"\nKeys: {len(repo_dict)}")      # There are 82 keys in repo_dict
# for key in sorted(repo_dict.keys()):
#     print(key)

# # The only way to know what information is available through an API is to
# # read the documentation or to examine the information through code, as we’re doing here

# # Let's pull out the values for some of the keys in repo_dict:

# # print("\nSelected information about first repository:")
# # print(f"Name: {repo_dict['name']}")
# # print(f"Owner: {repo_dict['owner']['login']}")      # login - get the owner's login name
# # print(f"Stars: {repo_dict['stargazers_count']}")    # Number of stars earned
# # print(f"Repository: {repo_dict['html_url']}")
# # print(f"Created: {repo_dict['created_at']}")
# # print(f"Updated: {repo_dict['updated_at']}")
# # print(f"Description: {repo_dict['description']}")


# '''Summarize the Top Repositories'''

# # When we make a visualization for this data, we’ll want to include more than one repository. 
# # Let’s write a loop to print selected information about each repository the
# # API call returns so we can include them all in the visualization.

# print("\nSelected informtion about each repository.")
# for repo_dict in repo_dicts:
#     print("\nSelected information about first repository:")
#     print(f"Name: {repo_dict['name']}")
#     print(f"Owner: {repo_dict['owner']['login']}")      # login - get the owner's login name
#     print(f"Stars: {repo_dict['stargazers_count']}")    # Number of stars earned
#     print(f"Repository: {repo_dict['html_url']}")
#     print(f"Created: {repo_dict['created_at']}")
#     print(f"Updated: {repo_dict['updated_at']}")
#     print(f"Description: {repo_dict['description']}")


# '''Monitoring API Rate Limits'''

# # Most APIs have rate limits, which means there’s a limit to
# # how many requests you can make in a certain amount of time.

# # NOTE
# # Many APIs require you to register and obtain an API key or access token to make API calls.
# # As of this writing, GitHub has no such requirement, but if you obtain an access token,
# # your limits will be much higher.



'''Visualizing Repositories Using Plotly'''

# Let’s make a visualization using the data we’ve gathered to show the relative
# popularity of Python projects on GitHub.
# We’ll make an interactive bar chart: the height of each bar will represent 
# the number of stars the project has acquired, and you’ll be able to click 
# the bar’s label to go to that project’s home on GitHub.


import requests
import plotly.express as px

# Make an API call and check the response.
url = "https://api.github.com/search/repositories"
url += "?q=language:python+sort:stars+stars:>10000"
headers = {"Accept": "application/vnd.github.v3+json"}
r = requests.get(url, headers=headers)
print(f"Status code: {r.status_code}")

# Process overall results.
response_dict = r.json()
print(f"Complete results: {not response_dict['incomplete_results']}")

# Process repository information.
repo_dicts = response_dict['items']
repo_names, stars = [], []
for repo_dict in repo_dicts:
    repo_names.append(repo_dict['name'])
    stars.append(repo_dict['stargazers_count'])

# Make visualisation.
fig = px.bar(x=repo_names, y=stars)
fig.show()
