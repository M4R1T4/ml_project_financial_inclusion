## Machine Learning Project - Financial Inclusion in Africa

![](/plots_and_pictures/africa_symbol_big.png)

We are [SaliFishe](https://github.com/SaliFishe) and 
[M4R1T4](https://github.com/M4R1T4) and we both enjoy Data Analysis and Machine Learning. We teamed up to work on this [Zindi Project](https://zindi.africa/competitions/financial-inclusion-in-africa) , explore  the Data, create Machine Learning Models and have fun learning while doing it. 



## The Zindi-Project

Financial inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 14% of adults) have access to or use a commercial bank account

Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and make payments while also helping businesses build up their credit-worthiness and improve their access to loans, insurance, and related services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.

The objective of this competition is to create a machine learning model to predict which individuals are most likely to have or use a bank account. The models and solutions developed can provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania and Uganda, while providing insights into some of the key factors driving individuals’ financial security.

## Our Repository ...

... is still in progress.  
... but feel free to take a look at our previous work:

### Structure and Content

| Folder| Content| Description|
|---|---|---|
| main| financial_inclusion_eda.md | Presentation and description of our insights from the data |
|data_analysis| 01_Data_and_EDA.ipynb | Jupyter notebook with the code to create mainly the bar plots and heatmaps (author M4R1T4)|
|models|01_Baseline_Model.ipynb| The KNN Base Model (author M4R1T4)|
|models|02_KNN_Model_Scaling_CrossVal.ipynb| KNN Scaling Cross Validation (author M4R1T4)|
|models|03_KNN_Model_Variants.ipynb| KNN Models with different parameter (author M4R1T4)|
|models|04_Tree_Models_Variants.ipynb| Decision Tree and Random Forest Models (author M4R1T4)|
|models| model_functions.py| functions for modelling and evaluation (author M4R1T4)|
|models| model_overview.md| Overview of the performance of the different models (author M4R1T4)|
---

## Set up your Environment



### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```
