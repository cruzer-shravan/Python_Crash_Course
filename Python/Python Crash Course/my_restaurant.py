"""This class represents the blueprint of a restaurant"""

class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0

    def describe_restaurant(self):
        print(f"\nThe restaurant's name is {self.restaurant_name.title()}.")
        print(f"{self.restaurant_name.title()} hosts {self.cuisine_type.title()} cuisine.")
        # print(f"The number of customers the restaurant has served so far is {self.number_served}.")
        # print(f"The restaurant served an additional {self.last_increment} customers, totalling to {self.total_served}.")   # added due to increment
    
    def open_restaurant(self):
        print(f"The {self.restaurant_name} restaurant is OPEN.")

    def set_number_served(self, number_served):                         # Addition of update_odometer
        """Sets the number of customers served to new value."""
        self.number_served = number_served
    
    def increment_number_served (self, additional_increment):
        self.total_served = self.number_served + additional_increment
        self.last_increment =  additional_increment
