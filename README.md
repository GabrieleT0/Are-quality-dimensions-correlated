# Are quality dimensions correlated? 🔎
This repository contains the code to reproduce the results presented in the publication *Are quality dimensions correlated? An empirical investigation*, accepted as research paper to ISWC 2025. From the following GitHub pages you can directly view the corrrelation matrices (as heatmaps) for each tool and summary tables for the 4 tools analyzed: 

In the folder [/data/inferential_statistics_results/](./data/inferential_statistics_results/), the correlation matrices are already available as CSV files. Meanwhile, in [/data/charts/heatmap](./data/charts/heatmap/), heatmaps created from the CSV files are provided for a graphical visualization of the correlation matrices.

If, instead, you want to regenerate the matrices and graphs using ther provided quality data, refer to section [How to reproduce the correlation matrices](#how-to-reproduce-the-correlation-matrices-).

# How to reproduce the correlation matrices 🔬
Download the quality data computed by KGHeartBeat here: [https://drive.google.com/file/d/10oY0Vk-fdhzjlDoHE9_BHx-6M3mUrrS3/view?usp=sharing](https://drive.google.com/file/d/10oY0Vk-fdhzjlDoHE9_BHx-6M3mUrrS3/view?usp=sharing). Move the zip file downloaded into [/data/quality_analysis_results](./data/quality_analysis_results/).

## Run the code 🚀
**Requirements**
- Python 3.13 or later.
- pip installed on your system.
- zip on linux to extract compressed files (on Windows, any software capable of decompressing a .zip file is suitable).
### Linux and MacOS users
A [shell script](run_correlation_analysis.sh) has been created to simplify and expedite the process of reproducing the results.
Make sure to grant execution permission to the script.
```sh
chmod +x run_correlation_analysis.sh
```

Then, execute it
```sh
./run_correlation_analysis.sh
```
This script will create and activate a Python virtual environment, install the required dependencies using ```pip install```, and extract the files containing the KGHeartBeat quality data.
At the end, the ```main.py``` file is executed, which calculates the correlation on the quality data from the four tools. The correlation matrices will be saved in [/data/inferential_statistics_results/](./data/inferential_statistics_results/), while the heatmaps will be stored in [/data/charts/heatmap](./data/charts/heatmap/)

### Windows users

1. Create and activate a Python Virtual Environment (Optional but Recommended)
```sh
# Create a Python venv
python -m venv venv
# Activate it
.\venv\Scripts\activate
```

2. Install the requirements
```sh
pip install -r requirements.txt
```

3. Unzip the KGHeartBeat quality data with any software capable of decompressing a .zip file is suitable. The file to be decompressed are located in the folder [data/quality_analysis_results](./data/quality_analysis_results)

4. Run the main.py file

```sh
python main.py
```
