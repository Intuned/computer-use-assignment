# type: ignore
def print_stuff(prefix, *args, **kwargs):
    # pass
    print(f"[{prefix}]", args, kwargs)


def noop(*_, **__):
    pass
