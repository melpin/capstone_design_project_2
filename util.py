def checkNone(value):
    """
    Check if string is None
    """
    if not value:
        return 0
    if value == 'None':
        return 0

    return 1