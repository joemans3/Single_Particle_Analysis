"""
This is a utility module to house functions and classes involved in logging and printing to the console.

Mainly this is for beautifying the console output and logging to a file if needed

Might just be better to use pprint (url: https://docs.python.org/3/library/pprint.html)

"""


def beautiful_print(*args, **kwargs):
    """
    This function is a wrapper around the print function that adds a line of dashes before and after the print statement

    Parameters:
    -----------
    *args: list
        list of arguments to pass to the print function
    **kwargs: dict
        dict of keyword arguments to pass to the print function

    Returns:
    --------
    None
    """
    print("-" * 100)
    print(*args, **kwargs)
    print("-" * 100)
    return


def beautiful_print_dict(dict_to_print: dict, **kwargs):
    """
    This function is a wrapper around the print function that adds a line of dashes before and after the print statement

    Parameters:
    -----------
    dict_to_print: dict
        dict to print
    **kwargs: dict
        dict of keyword arguments to pass to the print function

    Returns:
    --------
    None
    """
    print("-" * 100)
    for key in dict_to_print.keys():
        print(f"{key}: {dict_to_print[key]}")
    print("-" * 100)
    return


def beautiful_print_list(list_to_print: list, **kwargs):
    """
    This function is a wrapper around the print function that adds a line of dashes before and after the print statement

    Parameters:
    -----------
    list_to_print: list
        list to print
    **kwargs: dict
        dict of keyword arguments to pass to the print function

    Returns:
    --------
    None
    """
    print("-" * 100)
    for i in range(len(list_to_print)):
        print(f"{i}: {list_to_print[i]}")
    print("-" * 100)
    return
