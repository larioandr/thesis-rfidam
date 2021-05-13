class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_err(x, y):
    """Get relative error between of y regarding x.
    """
    if abs(x) < 1e-10:
        return abs(y)
    return abs(x - y) / abs(x)


def fmt_err(x, y, max_err=.1, min_abs_val=.001):
    """Get error formatted with color.
    """
    err = get_err(x, y)
    if abs(x) > min_abs_val and err > max_err:
        color = bcolors.FAIL
    else:
        color = bcolors.OKGREEN
    return highlight(f'{err:.4f}', color)


def highlight(s, color):
    """Return a string with highlighted value.
    """
    return f'{color}{s}{bcolors.ENDC}'


def pluralize(n):
    return '' if n == 1 else 's'
