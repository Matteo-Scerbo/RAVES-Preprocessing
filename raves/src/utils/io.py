import re


def sanitize(s: str):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def validate_inputs(path: str):
    validate_obj(path)
    validate_mtl(path)
    validate_csv(path)


def validate_obj(path: str):
    # TODO
    pass


def validate_mtl(path: str):
    # TODO
    pass


def validate_csv(path: str):
    # TODO
    pass
