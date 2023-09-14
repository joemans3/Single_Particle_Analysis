Make sure you have anaconda installed:
https://www.anaconda.com/download

Now follow these steps:
1) download or clone this repository.
2) In the conda prompt, navigate to the folder where you downloaded this repository using : cd <path to folder>
3) Using the SMT_env_BP.yml file, create a new environment using: conda env create -f SMT_env.yml
4) Activate the environment using: conda activate SMT_env_BP
5) Install the extra pip packages using: pip install -r requirements.txt
6) Since tensflow and tensorflow-probability are platform dependent we need to install inidividually.
i) Try the conda install method: conda install tensorflow, conda install tensorflow-probability
ii) If the above method fails, try the pip install method: pip install tensorflow, pip install tensorflow-probability

7) TODO add a test script to check if the installation is successful.


