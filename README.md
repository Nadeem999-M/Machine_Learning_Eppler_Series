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
## 2️⃣ Data Cleaning and Preprocessing

Once the raw data was successfully loaded into the structured format (`dataStruct`), the next step was to clean and prepare it for further analysis and model development.

---

## 🎯 Objective of this Stage

- Remove redundant rows and columns from the dataset.
- Ensure only relevant performance parameters are retained.
- Handle missing data by removing rows with incomplete records.

---

## 📂 Process Summary

- **Input:** `Workspace_Raw_Data.mat` (created in Section 1)
- **Main Dataset:** Performance data extracted from `.csv` files.
- **Output:** Cleaned `dataStruct` with properly formatted tables.

---

## 🔧 Code Explanation

### 1️⃣ Load Raw Data

```matlab
% Load the previously created data structure
load("Workspace_Raw_Data.mat")
```

### 2️⃣ Remove Redundant Rows and Columns

- The original performance data contained extra header rows and unnecessary columns.
- Keep only rows 7 and onward.
- Retain only the first 8 columns: Alpha, Cl, Cd, CDp, Cm, Top Xtr, Bot Xtr, Cpmin.

```matlab
% Remove rows from 1 to 6 from Main_Dataset and keep the first 8 columns
for idx = 1:length(dataStruct)
    if isfield(dataStruct(idx), 'Main_Dataset') && ...
       isfield(dataStruct(idx).Main_Dataset, 'data') && ...
       ~isempty(dataStruct(idx).Main_Dataset.data)

        C = dataStruct(idx).Main_Dataset.data;  % Cell array, 1st row = headers

        % Remove rows 1 to 6 and keep only the first 8 columns
        filteredData = C(7:end, 1:8);  % Keep rows from 7 onwards and first 8 columns

        % Convert to table and label the columns
        dataTable = cell2table(filteredData, 'VariableNames', {'Alpha', 'Cl', 'Cd','CDp','Cm','Top Xtr','Bot Xtr','Cpmin'});

        % Store the table back into the struct
        dataStruct(idx).Main_Dataset.data = dataTable;
    end
end
```

### 3️⃣ Handle Missing Values

- Remove any rows that contain missing (empty or NaN) values to ensure clean data for ML training.

```matlab
% Check for any missing values and remove them
for idx = 1:length(dataStruct)
    if isfield(dataStruct(idx), 'Main_Dataset') && ...
       isfield(dataStruct(idx).Main_Dataset, 'data') && ...
       ~isempty(dataStruct(idx).Main_Dataset.data)

        dataTable = dataStruct(idx).Main_Dataset.data;  % Table format

        % Remove rows with any missing values
        dataTable = rmmissing(dataTable);

        % Store the cleaned table back into the struct
        dataStruct(idx).Main_Dataset.data = dataTable;
    end
end
```

---

## ✅ Output of This Stage

- The original messy raw data is now fully cleaned and stored inside `dataStruct`.
- Each airfoil case contains only valid, clean, and ready-to-use performance data.

---
## 3️⃣ Exploratory Data Analysis (EDA)

After cleaning the dataset, exploratory data analysis was performed to better understand the relationships between different performance parameters of the airfoils.

---

## 🎯 Objective of this Stage

- Visualize correlation between features.
- Identify strong and weak relationships among the extracted aerodynamic coefficients.
- Gain insights to support feature engineering and model training.

---

## 📂 Process Summary

- **Input:** Cleaned `dataStruct` from Section 2
- **Analysis Tool:** Correlation Matrix and Heatmap visualization
- **Output:** Correlation plot that shows the strength of relationships between parameters.

---

## 🔧 Code Explanation

### 1️⃣ Variable Preparation

We selected the following features for correlation analysis:

- Alpha (Angle of Attack)
- Cl (Lift Coefficient)
- Cd (Drag Coefficient)
- CDp (Profile Drag Coefficient)
- Cm (Moment Coefficient)
- Top Xtr (Upper Surface Transition)
- Bot Xtr (Lower Surface Transition)
- Cpmin (Minimum Pressure Coefficient)

```matlab
% Define variable names for columns 1 to 8
varNames = {'Alpha', 'Cl', 'Cd', 'CDp', 'Cm', 'Top Xtr', 'Bot Xtr', 'Cpmin'};
```

---

### 2️⃣ Aggregate Data from All Airfoils

We aggregated all the cleaned data across the full dataset into a single array for correlation calculation.

```matlab
% Aggregate numeric data from all datasets (columns 1 to 8)
allData = [];
for idx = 1:length(dataStruct)
    if ~isempty(dataStruct(idx).Main_Dataset.data)
        T = dataStruct(idx).Main_Dataset.data;
        T = T(:,1:8); % Restrict to columns 1-8
        numericVars = varfun(@isnumeric, T, 'OutputFormat', 'uniform');
        allData = [allData; T{:, numericVars}];
    end
end
```

---

### 3️⃣ Generate Correlation Matrix & Heatmap

Finally, we computed the correlation coefficients and visualized them using MATLAB’s heatmap function.

```matlab
% Compute correlation matrix
R = corr(allData, 'Rows', 'complete');

% Create heatmap with custom axis labels
figure;
h = heatmap(varNames, varNames, R);

% Add title and axis labels
h.Title = 'Combined Correlation Matrix (Columns 1 to 8)';
h.XLabel = 'Variables';
h.YLabel = 'Variables';
```

---

## ✅ Output of This Stage

- Heatmap displaying pairwise correlation between all features.
- Helps to understand multicollinearity and feature importance for ML model development.

---
## 4️⃣ Feature Engineering & GUI Plotting

In this stage, we added important features to the dataset and developed an interactive MATLAB GUI for data visualization.

---

## 🎯 Objectives

- Create additional aerodynamic features to improve model performance.
- Provide user-friendly visualization of airfoil performance and geometry.

---

## 📂 Process Summary

### 1️⃣ Feature Engineering

- Calculated new feature: **Cl/Cd** ratio (Lift-to-Drag Ratio).
- Added this as an extra column in the dataset for each airfoil.

```matlab
% Calculate Cl/Cd and add to the dataset
for idx = 1:length(dataStruct)
    if isfield(dataStruct(idx), 'Main_Dataset') && isfield(dataStruct(idx).Main_Dataset, 'data') && ~isempty(dataStruct(idx).Main_Dataset.data)
        dataTable = dataStruct(idx).Main_Dataset.data;

        Cl = dataTable.Cl;
        Cd = dataTable.Cd;

        ClCd = Cl ./ Cd;
        ClCd(Cd == 0) = NaN; % Avoid division by zero

        dataTable.ClCd = ClCd;

        dataStruct(idx).Main_Dataset.data = dataTable;
    end
end
```
2️⃣ GUI Plotting
An interactive MATLAB GUI was created to visualize:
- Cl vs Cd
- Cl vs Alpha
- Cd vs Alpha
- Cl/Cd vs Alpha
- Cl & Cd vs Alpha (superimposed dual-axis plot)
- Airfoil Geometry (X vs Y)

The GUI allows the user to select:
- Airfoil + Reynolds Number combination
- Type of plot

The plots are then generated accordingly.

✅ Key code used to launch GUI:
```
% Call the airfoilPlotterGUI function with the processed dataStruct
airfoilPlotterGUI(dataStruct)
```
✅ Main GUI logic (simplified structure):
```
% Create main figure window with dropdown menus for airfoil and plot type
f = figure('Name', 'Airfoil Plotter');

% Dropdown list of Airfoil + Re combinations
airfoilReList = arrayfun(@(s) sprintf('%s_Re=%d', string(s.Airfoil), s.Reynolds), dataStruct, 'UniformOutput', false);
popupAirfoil = uicontrol(f, 'Style', 'popupmenu', 'String', airfoilReList);

% Dropdown list of plot options
popupPlot = uicontrol(f, 'Style', 'popupmenu', 'String', {'Cl vs Cd', 'Cl vs Alpha', 'Cd vs Alpha', 'Cl/Cd vs Alpha', 'Geometry (X vs Y)', 'Cl & Cd vs Alpha'});

% Plot Button executes the plotting based on user selection
uicontrol(f, 'Style', 'pushbutton', 'String', 'Plot', 'Callback', @(~,~) plotSelected(dataStruct, popupAirfoil, popupPlot, ax));
```
**✅ Output of This Stage**

- Fully cleaned dataset with additional Cl/Cd feature.
- Interactive GUI for visualizing all key aerodynamic relationships and airfoil geometry.
---
## 5️⃣ Coordinate Geometry Flattening

In this stage, we incorporated **airfoil geometry coordinates** directly into the performance dataset to prepare a complete dataset for machine learning.

---

## 🎯 Objectives

- Convert 2D geometry coordinate data into 1D flattened vectors.
- Merge the flattened geometry data into the main dataset for each airfoil-Reynolds combination.

---

## 📂 Process Summary

### 1️⃣ Flatten Geometry Coordinates

- Original geometry files contain columns of X and Y coordinates.
- We flatten the coordinates into one row:  
  `[x1, x2, ..., xn, y1, y2, ..., yn]`.

- This flattened geometry is repeated for every corresponding aerodynamic record in the dataset to match their dimensions.

### 2️⃣ Insert Flattened Geometry into Dataset

- The flattened geometry was inserted into each data table after the 9th column (after `Cl/Cd` column).
- Geometry variables were named as:  
  `x1, x2, ..., xn, y1, y2, ..., yn` (where `n` is the number of geometry points, e.g., 61).

---

## ✅ Key Code Logic

```matlab
for i = 1:length(dataStruct)
    entry = dataStruct(i);

    % Skip if missing geometry or main dataset
    if isempty(entry.Coordinate_Geometry.data) || isempty(entry.Main_Dataset.data)
        continue;
    end

    % Flatten geometry into row vector
    geom = entry.Coordinate_Geometry.data;
    x = geom(:, 1)';
    y = geom(:, 2)';
    geom_flat = [x, y];

    % Convert main dataset to table
    raw = entry.Main_Dataset.data;
    if iscell(raw)
        headers = string(raw(1, :));
        dataRows = raw(2:end, :);
        T = cell2table(dataRows, 'VariableNames', headers);
    elseif istable(raw)
        T = raw;
    else
        warning("Unrecognized format");
        continue;
    end

    % Create geometry table matching T rows
    nRows = height(T);
    geomTable = array2table(repmat(geom_flat, nRows, 1));
    nPoints = length(x);
    xVars = "x" + (1:nPoints);
    yVars = "y" + (1:nPoints);
    geomTable.Properties.VariableNames = [xVars, yVars];

    % Merge geometry columns into Main_Dataset
    T = [T(:,1:9), geomTable, T(:,10:end)];
    dataStruct(i).Main_Dataset.data = T;
end
```
✅ Output of This Stage

- Each aerodynamic record now includes its corresponding geometry.
- The dataset is now fully ready for ML preprocessing.

---
## 📧 Contact
Email: mn542052@gmail.com
**Project Author:**  
Mohammed Nadeem

---

> _This repository serves as full documentation and codebase for Eppler Airfoil Analysis and Machine Learning Model project._

