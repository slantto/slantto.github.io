%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the Serial port connected to the SMART Robot sensor board %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [S_Logger] = Init_Logger(ComPortNumber)
 
port = strcat('COM',num2str(ComPortNumber) );
 
out = instrfind('Port', port);  % Check to see if THAT serial port is already defined in MATLAB
if (~isempty(out))  % It is
    disp('WARNING:  port in use.  Closing.')
    if (~strcmp(get(out(1), 'Status'),'open'))  % Is it open?
        delete(out(1)); % If not, delete
    else  % is open
        fclose(out(1));
        delete(out(1)); 
    end
end
 
S_Logger = serial(port);            % Define serial port
set(S_Logger, 'BaudRate', 115200);  % Set the Baud rate to 115200
set(S_Logger, 'Tag', 'Logger');     % give it a name
set(S_Logger, 'TimeOut', .1);
% want to make the buffer big enough that new message can always fit
% The message size is 50 bytes
set(S_Logger, 'InputBufferSize',75);
fopen(S_Logger);                    % Open it;
pause(1)                            % give it a second to start getting data
disp('Data Logger Initialized')


