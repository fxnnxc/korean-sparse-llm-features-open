__all__ = [
    'GeneralFetch',
    'MultipleFetch',
]


class GeneralFetch:

    def __init__(self, module, condition_fn, fetch_fn):
        self.module = module
        self.condition_fn = condition_fn
        self.fetch_fn = fetch_fn
        self.can_fetch = True
        self.is_hook_registered = False
        self.register_hook()

    def get_fetch_hook(self):
        def fetch_hook(module, input, output):
            if self.can_fetch and self.condition_fn(module, input, output):
                return self.fetch_fn(module, input, output)
            else:
                return output
        return fetch_hook

    def register_hook(self, warn=True):
        if self.is_hook_registered:
            if warn:
                print("Warning: Hook is already registered, we remove the existing hook")
            self.remove_hook()
        self.fetch_hook = self.module.register_forward_hook(self.get_fetch_hook())
        self.is_hook_registered = True

    def remove_hook(self):
        self.fetch_hook.remove()
        self.is_hook_registered = False

    def deactivate(self):
        assert self.is_hook_registered, "Hook is not registered"
        self.can_fetch = False

    def activate(self):
        assert self.is_hook_registered, "Hook is not registered"
        self.can_fetch = True


class MultipleFetch:

    def __init__(self, dict_format):
        self.fetches = {
            #  "mlp": {
            #      "module": model.transformer.h[10].mlp,
            #      "condition_fn": lambda module, input, output: True,
            #      "fetch_fn": lambda module, input, output: output
            #  },
            #  "attn": {
            #      "module": model.transformer.h[10].attn,
            #      "condition_fn": lambda module, input, output: True,
            #      "fetch_fn": lambda module, input, output: output
            #  }
        }
        for key, value in dict_format.items():
            fetch = GeneralFetch(value["module"], value["condition_fn"], value["fetch_fn"])
            self.fetches[key] = fetch

    def activate(self, key=None):
        if key is None:
            for key in self.fetches.keys():
                self.fetches[key].activate()
        else:
            self.fetches[key].activate()

    def deactivate(self, key=None):
        if key is None:
            for key in self.fetches.keys():
                self.fetches[key].deactivate()
        else:
            self.fetches[key].deactivate()

    def remove_hook(self, key=None):
        if key is None:
            for key in self.fetches.keys():
                self.fetches[key].remove_hook()
        else:
            self.fetches[key].remove_hook()

    def register_hook(self, key=None):
        if key is None:
            for key in self.fetches.keys():
                self.fetches[key].register_hook()
        else:
            self.fetches[key].register_hook()
