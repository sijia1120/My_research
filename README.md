# My_research

(1) `total_macro.csv` is the 135 macroeconomic variables (127 from current.csv + 8 from PredictorData2021.xlsx) [from 1980-02 to 2021-12]

(2) 'current_transformed.csv' is the transformation of original `current.csv` [from 1959-01 to 2023-12]

(3) 'macro8_transformed.csv' is the transformation of original 'PredictorData2021.xlsx' [from 1871-01 to 2021-12]

(4) `Macro_data_comparasion.xlsx` describe 135 Macroeconomic variables (Description / min/max/mean/std/correlation); correlation of 127 variables is between 'current_transformed.csv' and 'Macro.csv' which is downloaded from Markus's website. correlation of 8 variables is between 'macro8_transformed.csv' and ''Macro.csv' which is downloaded from Markus's website. 

(5) `total_macro_normalized.csv` is normalized dataset of `total_macro.csv`. The normalization is rank norm which is shown in `3_Data_preprocess.ipynb`
