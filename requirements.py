import os
import subprocess
import pipreqs



def create_requirements_with_pipreqs():
    current_dir = os.getcwd()

    try:
        # Use pipreqs to create requirements.txt
        subprocess.run(['pipreqs', current_dir, '--force'], check=True)
        print("requirements.txt created successfully using pipreqs.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_requirements_with_pipreqs()
