def make_pizza(size, *toppings):
    """Describes the size of the pizza along with requested toppings."""
    print(f"\nMaking {size}-inch pizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping.title()}")
