"""These classes represent the user, admin and privileges allotted to the admin"""

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


class Admin(User):
    """Represents the privileges provided to Admin."""
    def __init__(self, first_name, last_name, age, occupation):
        super().__init__(first_name, last_name, age, occupation)
        self.privileges = Privileges()   # store object, not list
    
    def show_privileges(self):
        """Displays the set of privileges provided to the Administrator."""
        print("The adminstrator has the following privileges:")
        for privilege in self.privileges:
            print(f"- {privilege.title()}")


class Privileges:
    """A class to store admin's privileges"""

    def __init__(self, privileges = None):
        if privileges is None:
            # self.privileges_list = []
            print("This user has no privileges.")
        else:
            self.privileges_list = privileges
    
    def show_privileges(self):
        """Displays the set of privileges provided to the Administrator."""
        # print(f"The adminstrator {} has the following privileges:")
        for privilege in self.privileges_list:
            print(f"- {privilege.title()}")
