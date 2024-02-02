from langchain.tools import tool

@tool
def read_file(file_path:str):
    """Read file from local directory"""

    with open(file_path, 'r') as f:
            return "File contents: " + f.read()
    
@tool
def write_file(file_path:str, contents:str):
    """Write file to local directory"""

    with open(file_path, 'w') as f:
        f.write(contents)
        return "File '" + str(file_path) + "' saved."