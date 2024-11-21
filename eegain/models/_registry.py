all_models = {}


def register_model(cls):
    model_name = cls.__name__
    all_models[model_name] = cls
    return cls


def list_models():
    return all_models.keys()
