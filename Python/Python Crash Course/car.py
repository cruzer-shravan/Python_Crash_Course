#1 """A class that can be used to represent a car."""
"""A set of classes used to represent gas and electric cars."""

class Car:
    """A simple attempt to represent a car"""

    def __init__(self, make, model, year):
        """Initialise attributes to describe a car."""
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0       # Python created a new attribute called odometer_reading and sets its initial value to 0.

    def get_descriptive_name(self):
        """Return a neatly formatted descriptive name."""
        long_name = f"{self.year} {self.make.title()} {self.model.title()}"
        return long_name.title()
    
    def read_odometer(self):
        "Print a statement showing the car's mileage."
        print(f"This car has {self.odometer_reading} miles on it.")
    
    def update_odometer(self, mileage):                         # Addition of update_odometer
        """Set the odometer reading to the given value.
        Reject the change if it attempts to roll back the odometer."""
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can't roll back on odometer.")
    
    def increment_odometer(self, miles):
        """Add the given amount to the odometer reading."""
        self.odometer_reading += miles


# class Battery:
#     """A simple attempt to model a battery for an electric car."""

#     def __init__(self, battery_size = 40):
#         """Initialize the battery's attributes."""
#         self.battery_size = battery_size
    
#     def describe_battery(self):
#         """Print a statement describing the battery size."""
#         print(f"The car has a {self.battery_size} kWh battery.")
    
#     def get_range(self):
#         """Print a statement about the range this battery provides."""
#         if self.battery_size == 40:
#             range = 150
#         elif self.battery_size == 65:
#             range = 225
        
#         print(f"This car has a range of {range} miles on a full charge.")
    
#     def upgrade_battery(self):
#         if self.battery_size != 65:
#             self.battery_size = 65


# class ElectricCar(Car):
#     """Initializes attributes of parent class."""
#     """Then initialize attributes specific to an electric car."""
#     def __init__(self, make, model, year):
#         super().__init__(make, model, year)
#         self.battery = Battery()                  # Calling the instance from the class Battery

#     def describe_battery(self):
#         """Print a statement describing the battery size."""
#         print(f"This car has a {self.battery_size} kWh battery.")