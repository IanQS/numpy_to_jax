import os 

def rename_folder(proj_name):
    """
    Set up the folder under which you will create your package files
    """
    full_path = os.path.realpath(__file__)
    path, _= os.path.split(full_path)
    os.rename(f"{path}/base_package_name", f"{path}/{proj_name}")

def change_main_import(proj_name):
    full_path = os.path.realpath(__file__)
    path, _= os.path.split(full_path)
    with open(f"{path}/main.py") as f:
        lines = f.readlines()

    splitted = lines[0].split()
    splitted[1] = proj_name
    lines[0] = " ".join(splitted)

    with open(f"{path}/main.py", "w") as f:
        f.writelines(lines)


def setup_conda(proj_name):
    """
    Set up the conda environment name
    """

    full_path = os.path.realpath(__file__)
    path, _= os.path.split(full_path)
    with open(f"{path}/environment.yml") as f:
        lines = f.readlines()

    lines[0] = f"name: {proj_name}\n"

    with open("environment.yml", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    proj_name = input("What is the name of your project? Use \'_\' instead of spaces ")
    rename_folder(proj_name)
    setup_conda(proj_name)
    change_main_import(proj_name)


