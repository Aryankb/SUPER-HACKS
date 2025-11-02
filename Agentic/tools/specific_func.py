import functools

# Global registry of tools by user
USER_TOOLS = {}

def user_tool(user_id: str):
    """
    Decorator to register a tool for a specific user.
    """
    def decorator(func_or_class):
        # Register tool under this user
        USER_TOOLS.setdefault(user_id, []).append(func_or_class)

        @functools.wraps(func_or_class)
        def wrapper(*args, **kwargs):
            # Inject user_id if function supports it
            if "user_id" in func_or_class.__code__.co_varnames:
                kwargs["user_id"] = user_id
            return func_or_class(*args, **kwargs)
        
        return wrapper
    return decorator


