class Product:
    inventory = []

    def __init__(self, product_id, name, category, quantity, price, supplier):
        self.product_id = product_id
        self.name = name
        self.category = category
        self.quantity = quantity
        self.price = price
        self.supplier = supplier

    @classmethod
    def add_product(cls, name, category, quantity, price, supplier):
        # Auto-generate product_id incrementally with no duplicates
        if cls.inventory:
            product_id = max(p.product_id for p in cls.inventory) + 1
        else:
            product_id = 1

        new_product = Product(product_id, name, category, quantity, price, supplier)
        cls.inventory.append(new_product)
        return "Product added successfully"

    @classmethod
    def update_product(cls, product_id, quantity=None, price=None, supplier=None):
        for product in cls.inventory:
            if product.product_id == product_id:
                if quantity is not None:
                    product.quantity = quantity
                if price is not None:
                    product.price = price
                if supplier is not None:
                    product.supplier = supplier
                return "Product information updated successfully"
        return "Product not found"

    @classmethod
    def delete_product(cls, product_id):
        for product in cls.inventory:
            if product.product_id == product_id:
                cls.inventory.remove(product)
                return "Product deleted successfully"
        return "Product not found"


class Order:
    def __init__(self, order_id, products, customer_info=None):
        self.order_id = order_id
        self.products = products
        self.customer_info = customer_info

    def place_order(self, product_id, quantity, customer_info=None):
        self.products.append((product_id, quantity))
        if customer_info is not None:
            self.customer_info = customer_info
        # Reduce the product's inventory quantity
        for product in Product.inventory:
            if product.product_id == product_id:
                product.quantity -= quantity
                break
        return f"Order placed successfully. Order ID: {self.order_id}"
