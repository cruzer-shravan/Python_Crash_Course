"""This class represent the details about the user."""

class User:
    def __init__(self, first_name, last_name, age, occupation):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.occupation = occupation
        self.login_attempts = 0
    
    def describe_user(self):
        user_info = f"Name: {self.first_name.title()} {self.last_name.title()}, \n\tAge: {self.age}, \n\tOccupation: {self.occupation.title()}, \n\tlogin attempts: {self.login_attempts}"
        print("The description of the user is appended below: ")
        print(f"User info: {user_info}")
    
    def greet_user(self):
        full_name = f"{self.first_name} {self.last_name}"
        print(f"\nHello, {full_name.title()}!")
    
    def increment_login_attempts(self):
        self.login_attempts += 1
    
    def reset_login_attempts(self):
        self.login_attempts = 0
