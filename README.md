# Machine_Learning_Eppler_Series
## Project Overview

This project focuses on using machine learning to analyze and generate Eppler series airfoil designs. The workflow consists of several stages: data preparation, data processing, model training, and model deployment using MATLAB and XFLR5.

---
## 1️⃣ Data Collection and Initial Structuring

The first step in this project was to collect raw data for the Eppler series airfoils. The airfoil geometry and aerodynamic performance data were organized into a systematic structure for further processing.

### 📂 Dataset Overview

- **Source Website:** [airfoiltools.com](https://airfoiltools.com/)
- **Airfoil Series:** Eppler
- **Total Airfoils:** 192
- **Geometry Files:** `.dat` files (coordinate files)
- **Performance Files:** `.csv` files (performance data extracted from XFLR5)

### 📌 Task Goals for This Section:

- Load all `.dat` geometry files.
- Load corresponding `.csv` performance files for each Reynolds number.
- Store all data into a MATLAB structure `dataStruct` for easy access and processing.

---

### 🔧 Code Explanation

```matlab
% Set random seed for reproducibility
rng(2025)

% Define the folder path where all files are stored
folderPath = "C:\Users\Mohammed Nadeem\Desktop\Eppler Machine Learing Model\Airfoil Dataset";

% List all .dat files (coordinate geometry files)
dat_files = dir(fullfile(folderPath, '*.dat'));
airfoil_names = unique(erase({dat_files.name}, '.dat'));

% Reynolds numbers list
Re_list = [50000,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000];
nCases = length(Re_list);

% Preallocate struct array dynamically
dataStruct = struct();
idx = 1;

for a = 1:length(airfoil_names)
    airfoil = airfoil_names{a};
    coord_file = fullfile(folderPath, [airfoil '.dat']);
    
    % Read coordinate geometry once per airfoil
    try
        coord_data = readmatrix(coord_file);  % Geometry is numerical
    catch ME
        coord_data = [];
        warning('Could not read coordinate file %s\nError: %s', coord_file, ME.message);
    end
    
    for i = 1:nCases
        Re_num = Re_list(i);
        fprintf('\nProcessing Airfoil: %s | Re: %d\n', airfoil, Re_num);
        main_file = fullfile(folderPath, sprintf('%s_Re%d.csv', airfoil, Re_num));

        % Read main dataset as raw cell array
        try
            main_data = readcell(main_file, 'Delimiter', ',');
        catch ME
            main_data = [];
            warning('Could not read main file %s\nError: %s', main_file, ME.message);
        end

        % Store data into struct
        dataStruct(idx).Airfoil = airfoil;
        dataStruct(idx).Reynolds = Re_num;
        dataStruct(idx).Main_Dataset.file = main_file;
        dataStruct(idx).Main_Dataset.data = main_data;
        dataStruct(idx).Coordinate_Geometry.file = coord_file;
        dataStruct(idx).Coordinate_Geometry.data = coord_data;

        idx = idx + 1;
    end
end
```

---

✅ **Output of This Stage:**  
At the end of this section, a large `dataStruct` is created in MATLAB containing all the loaded airfoil geometries and aerodynamic performance data.

---

## 📧 Contact
Email: mn542052@gmail.com
**Project Author:**  
Mohammed Nadeem

---

> _This repository serves as full documentation and codebase for Eppler Airfoil Analysis and Machine Learning Model project._

