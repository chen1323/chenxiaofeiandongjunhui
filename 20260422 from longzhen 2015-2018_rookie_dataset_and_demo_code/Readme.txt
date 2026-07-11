README

Project
Replication files for the 2018 accounting rookies' performance prediction models.

Files included
1. main.py
   Main script used to run the prediction model and generate the output files.

2. 2015-2018_rookie_dataset.xlsx
   Input dataset used by the script.

3. output_prediction_main_2018.xlsx
   Model prediction output for the main 2018 specification.

4. output_accuracy_main_2018.xlsx
   Accuracy metrics corresponding to the main 2018 prediction output.

5. requirements.txt
   Python package requirements for reproducing the results with pip.

6. environment.yml
   Conda environment file for reproducing the results with conda.


Expected output
Running main.py should generate the main 2018 prediction results and the corresponding accuracy metrics.

If the script is configured exactly as provided, the expected output files are:
- output_prediction_main_2018.xlsx
- output_accuracy_main_2018.xlsx

Notes
- Please keep all files in the same folder unless you modify the file paths in main.py.
- The code is intended to reproduce the main 2018 result only. Other robustness analyses or alternative specifications are not included in this package.
- To ensure exact replication, use the provided environment file or requirements file rather than installing packages manually.