def show_tensor(name, value):
    print(f'{name}{tuple(value.size())}:\n {value.numpy()}')
