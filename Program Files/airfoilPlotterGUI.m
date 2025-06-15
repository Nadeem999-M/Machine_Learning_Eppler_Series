function airfoilPlotterGUI(dataStruct)
    % Create main figure
    f = figure('Name', 'Airfoil Plotter', ...
               'Position', [300, 300, 800, 500], ...
               'Resize', 'off');

    % Airfoil+Re list (e.g., "E216_Re=100000")
    airfoilReList = arrayfun(@(s) ...
        sprintf('%s_Re=%d', string(s.Airfoil), s.Reynolds), ...
        dataStruct, 'UniformOutput', false);

    % --- UI CONTROLS ---
    % Label: Airfoil + Re
    uicontrol(f, 'Style', 'text', 'String', 'Select Airfoil + Re:', ...
        'Position', [30 440 150 25], 'HorizontalAlignment', 'left', 'FontSize', 10);

    % Dropdown: Airfoil + Re
    popupAirfoil = uicontrol(f, 'Style', 'popupmenu', ...
        'String', airfoilReList, ...
        'Position', [180 445 250 25], ...
        'FontSize', 10);

    % Label: Plot Type
    uicontrol(f, 'Style', 'text', 'String', 'Select Plot Type:', ...
        'Position', [30 390 150 25], 'HorizontalAlignment', 'left', 'FontSize', 10);

    % Dropdown: Plot Type
    popupPlot = uicontrol(f, 'Style', 'popupmenu', ...
        'String', {'Cl vs Cd', 'Cl vs Alpha', 'Cd vs Alpha', 'Cl/Cd vs Alpha', 'Geometry (X vs Y)', 'Cl & Cd vs Alpha'}, ...
        'Position', [180 395 250 25], ...
        'FontSize', 10);

    % Axes for plotting (must be defined *before* callback)
    ax = axes('Parent', f, ...
              'Units', 'pixels', ...
              'Position', [100, 50, 650, 300]);

    % Plot Button
    uicontrol(f, 'Style', 'pushbutton', 'String', 'Plot', ...
        'Position', [450 420 100 30], ...
        'FontSize', 10, ...
        'Callback', @(~,~) plotSelected(dataStruct, popupAirfoil, popupPlot, ax));
end

function plotSelected(dataStruct, popupAirfoil, popupPlot, ax)
    idx = popupAirfoil.Value;
    plotChoice = popupPlot.Value;
    entry = dataStruct(idx);

    % Extract and convert 42x4 cell to table
    raw = entry.Main_Dataset.data;
    headers = string(raw.Properties.VariableNames);  % Get headers from table
    data = table2array(raw);                         % Convert table to array
    T = array2table(data, 'VariableNames', headers);

    % Clear current axes
    cla(ax);

    switch plotChoice
            case {1,2,3,4}  % Single y-axis plots (no yyaxis)
                cla(ax, 'reset');  % fully clear axes and remove right y-axis
                
                switch plotChoice
                    case 1  % Cl vs Cd
                        plot(ax, T.Cd, T.Cl, '-o');
                        xlabel(ax, 'Cd'); ylabel(ax, 'Cl');
                        title(ax, 'Cl vs Cd');
                    case 2  % Cl vs Alpha
                        plot(ax, T.Alpha, T.Cl, '-o');
                        xlabel(ax, 'Alpha (°)'); ylabel(ax, 'Cl');
                        title(ax, 'Cl vs Alpha');
                    case 3  % Cd vs Alpha
                        plot(ax, T.Alpha, T.Cd, '-o');
                        xlabel(ax, 'Alpha (°)'); ylabel(ax, 'Cd');
                        title(ax, 'Cd vs Alpha');
                    case 4  % Cl/Cd vs Alpha
                        plot(ax, T.Alpha, T.ClCd, '-o');
                        xlabel(ax, 'Alpha (°)'); ylabel(ax, 'Cl/Cd');
                        title(ax, 'Cl/Cd vs Alpha');
                end
        
            case 5  % Geometry plot (no yyaxis)
                cla(ax, 'reset');  % fully clear axes
                if isfield(entry, 'Coordinate_Geometry') && ...
                   isfield(entry.Coordinate_Geometry, 'data') && ...
                   ~isempty(entry.Coordinate_Geometry.data)
                    coordData = entry.Coordinate_Geometry.data;
                    X = coordData(:, 1);
                    Y = coordData(:, 2);
                    plot(ax, X, Y, '-r');
                    axis(ax, 'equal');  % allowed here, no yyaxis
                    xlabel(ax, 'X'); ylabel(ax, 'Y');
                    title(ax, 'Airfoil Geometry');
                else
                    msgbox('Geometry data not available for this airfoil.', 'Error', 'error');
                end
        
            case 6  % Cl and Cd vs Alpha with dual y-axis
                cla(ax, 'reset');  % fully clear axes before dual y-axis plot
                
                yyaxis(ax, 'left');
                plot(ax, T.Alpha, T.Cl, '-o', 'DisplayName', 'Cl', 'LineWidth', 1.5);
                ylabel(ax, 'Cl');
                
                yyaxis(ax, 'right');
                plot(ax, T.Alpha, T.Cd, '-s', 'DisplayName', 'Cd', 'LineWidth', 1.5, 'Color', [0.85 0.33 0.10]);
                ylabel(ax, 'Cd');
                
                xlabel(ax, 'Alpha (°)');
                title(ax, 'Cl and Cd vs Alpha');
                legend(ax, {'Cl', 'Cd'}, 'Location', 'Best');
                % Do NOT use axis equal or daspect here
    end

end