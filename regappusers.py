import re


def validate_name(name):
    if not isinstance(name, str) or len(name.strip()) < 2:
        raise ValueError("Name must be at least 2 characters long.")
    return True


def validate_email(email):
    if not isinstance(email, str) or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise ValueError("Invalid email format.")
    return True


def validate_password(password):
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    return True


def validate_user(name, email, password):
    validate_name(name)
    validate_email(email)
    validate_password(password)
    return True


def register_user(name, email, password):
    try:
        validate_user(name, email, password)
    except ValueError:
        return False

    return {"name": name, "email": email, "password": password}


# --- Quick demo ---
if __name__ == "__main__":
    # Successful registration
    result = register_user("Alice", "alice@example.com", "securepass")
    print("Success:", result)

    # Failed registration (bad email)
    result = register_user("Bob", "not-an-email", "securepass")
    print("Bad email:", result)

    # Failed registration (short password)
    result = register_user("Carol", "carol@example.com", "short")
    print("Bad password:", result)
