import yaml


def find_keys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in find_keys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in find_keys(j, kv):
                yield x


class Config:
    def __init__(self, path):
        with open(path, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def __getitem__(self, property_name):
        if "." in property_name:
            value = self.config
            for key in property_name.split('.'):
                value = value[key]
                if value is None:
                    raise Exception(f'Config: Not found value for {key} property of {property_name} path!')
            return value

        return self.config[property_name]

    def property(self, name):
        results = list(find_keys(self.config, name))

        if len(results) == 0:
            Exception(f'Not found {name} property!')
        elif len(results) > 1:
            Exception(f'More than one property with {name} name!')
