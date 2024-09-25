import os

def list_python_libraries(venv_lib_dir):
    if not os.path.exists(venv_lib_dir):
        print(f"Directory '{venv_lib_dir}' not found.")
        return

    # List all files and directories in the specified directory
    contents = os.listdir(venv_lib_dir)
    print("Contents of the directory:")
    libraries = []
    for item in contents:
        print(item)
        libraries.append(item)
    # sort the list alphabetically
    libraries = libraries.sort()

    #write libraries to a file called requirements.txt
    with open('requirements.txt', 'w') as f:
        for item in libraries:
            f.write("%s\n" % item)
        f.close();

if __name__ == "__main__":
    venv_lib_dir = '/Users/kevincowan/PycharmProjects/rockymtn_gpt2/.venv/lib/python3.12/site-packages'
    list_python_libraries(venv_lib_dir)