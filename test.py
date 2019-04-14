def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({"__file__": filepath, "__name__": "__main__"})
    with open(filepath, "rb") as f:
        exec(compile(f.read(), filepath, "exec"), globals, locals)


def run():
    locals = {}
    execfile("config.py", locals=locals)
    print(locals)
    print(locals["BATCH_SIZE"])


if __name__ == "__main__":
    run()
