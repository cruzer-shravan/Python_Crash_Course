"""These classes represent the details of an admin and the admin's privileges."""
from user import User

class Admin(User):
    """Represents the privileges provided to Admin."""
    def __init__(self, first_name, last_name, age, occupation):
        super().__init__(first_name, last_name, age, occupation)
        # self.privileges = Privileges()   # store object, not list
    
    def show_privileges(self):
        """Displays the set of privileges provided to the Administrator."""
        print("The adminstrator has the following privileges:")
        for privilege in self.privileges:
            print(f"- {privilege.title()}")


class Privileges(User):
    """A class to store admin's privileges"""

    def __init__(self, privileges = []):
        if privileges == []:
            # # self.privileges_list = []
            # # print("This user has no privileges.")
            # privilege = None
            print(f"The employee has no privileges.")
        else:
            self.privileges_list = privileges
            print(f"The adminstrator has the following privileges:")
            for privilege in self.privileges_list:
                print(f"- {privilege.title()}")
    
    # def show_privileges(self):
    #     """Displays the set of privileges provided to the Administrator."""
    #     print(f"The adminstrator has the following privileges:")
    #     for privilege in self.privileges_list:
    #         print(f"- {privilege.title()}")
