from kfp.v2.dsl import component


@component
def add(a: float, b: float) -> float:
    print(f'Adding {a} and {b} = {a + b}')
    return a + b
