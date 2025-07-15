"""Base tool implementation for the /tools directory."""

class Tool:
    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func
        