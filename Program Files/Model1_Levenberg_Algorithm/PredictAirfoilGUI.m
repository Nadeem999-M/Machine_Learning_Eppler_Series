function PredictAirfoilGUI
% PredictAirfoilGUI - Airfoil Prediction GUI using Neural Network

% Create figure window
f = figure('Name','Neural Network Airfoil Predictor','Position',[500 300 400 400]);

% Input labels and edit boxes
uicontrol(f,'Style','text','Position',[30 340 100 20],'String','Alpha:');
editAlpha = uicontrol(f,'Style','edit','Position',[150 340 200 25]);

uicontrol(f,'Style','text','Position',[30 300 100 20],'String','Cl:');
editCl = uicontrol(f,'Style','edit','Position',[150 300 200 25]);

uicontrol(f,'Style','text','Position',[30 260 100 20],'String','Cd:');
editCd = uicontrol(f,'Style','edit','Position',[150 260 200 25]);

uicontrol(f,'Style','text','Position',[30 220 100 20],'String','Cdp:');
editCdp = uicontrol(f,'Style','edit','Position',[150 220 200 25]);

% Predict button
btnPredict = uicontrol(f,'Style','pushbutton','String','Plot Geometry',...
    'Position',[50 150 130 40],'Callback',@predictCallback);

% Export button
btnExport = uicontrol(f,'Style','pushbutton','String','Export to Excel',...
    'Position',[220 150 130 40],'Callback',@exportCallback);

% Initialize empty coordinates (shared data)
coords = [];

% Callback function for prediction & plotting
    function predictCallback(~,~)
        try
            % Read input values
            Alpha = str2double(get(editAlpha,'String'));
            Cl = str2double(get(editCl,'String'));
            Cd = str2double(get(editCd,'String'));
            Cdp = str2double(get(editCdp,'String'));

            % Combine into input vector
            userInput = [Alpha; Cl; Cd; Cdp];

            % Call trained neural network function
            predictedOutput = myNeuralNetworkFunction(userInput);

            % Extract coordinates
            x_coords = predictedOutput(1:20);
            y_coords = predictedOutput(21:40);
            
            % Save coords for export
            coords = [x_coords(:), y_coords(:)];

            % Plot geometry
            figure('Name','Predicted Airfoil Geometry');
            plot(x_coords, y_coords, '-o','LineWidth', 2, 'MarkerSize',5)
            xlabel('X'); ylabel('Y');
            title('Predicted Airfoil Geometry');
            grid on; axis equal;
        catch err
            disp('Error during prediction. Check inputs or model.');
            disp(err.message)
        end
    end

% Callback function for export
    function exportCallback(~,~)
        if isempty(coords)
            msgbox('Please predict geometry first.','Error','error');
            return;
        end
        [file, path] = uiputfile('PredictedAirfoil.xlsx','Save As');
        if isequal(file,0)
            return; % Cancelled
        end
        filename = fullfile(path,file);
        writematrix(coords, filename);
        msgbox('Coordinates exported successfully.','Success','help');
    end

end