# Getting Started

## Linux (BASH)
### Github
Installing Git and verifying version:
- sudo apt-get update
- sudo apt-get install git
- git --version

Setting up git account
- git config --global user.name "Your Git Name"
- git config --global user.email "youremail@example.com"

Clone Github repository to your local machine (1) and navigate to the repository (2)
This assumes BASH context is set to the location where you want the repository to be placed
- git clone https://github.com/yourusername/my_project.git
- cd my_project

### Python
We will be using Python 3.11
3rd line includes the Virtual Environment package
- sudo add-apt-repository ppa:deadsnakes/ppa
- sudo apt-get update
- sudo apt-get install python3.11 python3.11-venv

Create Virtual Environment:
Navigate to your project directory (1) (or create using “mkdir”)
Create virtual environment (2), then activate (3)
- cd my_project
- python3.11 -m venv venv
- source venv/bin/activate

Install Packages:
Get requirements.txt from GitHub, place in project folder
With venv activated, install package
- pip install -r requirements.txt

## Windows (CMD)
### Github
Download and Install Git from https://git-scm.com/download/win
Verify git version:
- git --version

Setting up git account
- git config --global user.name "Your Git Name"
- git config --global user.email "youremail@example.com"

Navigate to the desired folder (1).
Clone Github repository to your local machine (2) and navigate to the repository (3)
- cd path\to\your\desired\folder
- git clone https://github.com/yourusername/my_project.git
- cd my_project

### Python
We will be using Python 3.11.9. When prompted during the Python install, check the box “Add Python to PATH” to ensure python and pip is accessible from the cmd.
Download and Install Python 3.11.9 https://www.python.org/downloads/release/python-3119/ 
Verify Python and pip version
- python --version
- pip --version

Create Virtual Environment:
Navigate to your project repository (1)
Create virtual environment (2), then activate (3)
- cd path\to\your\my_project
- python -m venv venv
- venv\Scripts\activate

Install Packages:
Get requirements.txt from GitHub, place in project folder
With venv activated, install package
- pip install -r requirements.txt
