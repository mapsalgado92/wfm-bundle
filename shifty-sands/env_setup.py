import os
import subprocess
import sys
import venv

def create_virtual_environment(venv_dir):
    """Create a virtual environment in the specified directory."""
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    print(f"Virtual environment created at {venv_dir}")

def install_dependencies(venv_dir):
    """Activate the virtual environment and install dependencies."""
    if os.name == 'nt':  # For Windows
        activate_script = os.path.join(venv_dir, 'Scripts', 'activate')
    else:  # For Unix or MacOS
        activate_script = os.path.join(venv_dir, 'bin', 'activate')

    # Install dependencies within the virtual environment
    pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip')
    subprocess.call([pip_executable, "install", "-r", "requirements.txt"])

    print("Dependencies installed.")

def activate_virtualenv(venv_dir):
    """Provide instructions to activate the virtual environment."""
    if os.name == 'a':
        activate_command = f"{venv_dir}\\Scripts\\activate"
    else:  # For Unix or MacOS
        activate_command = f"source {venv_dir}/bin/activate"

    print(f"To activate the virtual environment, run:\n\n{activate_command}\n")

def main():
    venv_dir = '.venv'
    
    if not os.path.exists(venv_dir):
        create_virtual_environment(venv_dir)
    
    install_dependencies(venv_dir)
    activate_virtualenv(venv_dir)

if __name__ == "__main__":
    main()