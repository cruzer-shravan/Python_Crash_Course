from car import Car
from my_electric_car import ElectricCar

my_mustang = Car('ford', 'mustang', 2025)
print(my_mustang.get_descriptive_name())

my_leaf = ElectricCar('nissan', 'leaf', 2025)
print(my_leaf.get_descriptive_name())
