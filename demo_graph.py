import os
from graph import UnitTestGraph

def run_graph():

    graph = UnitTestGraph()

    code = """
        class Test():
            print("Howdy!")

    """

    input_values = {"code": code}

    graph.invoke(input_values=input_values)

if __name__ == "__main__":
    
    run_graph()