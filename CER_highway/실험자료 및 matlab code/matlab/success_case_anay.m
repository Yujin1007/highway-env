reload  = true;
close all
if reload
    clear;

    xlRange = 'A2:F101';
    cer = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/CER.xlsx';
    cerns = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/CERns.xlsx';
    td = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/TD3.xlsx';
    tdns = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/TDns.xlsx';
    
    cermpc = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/MPCCER.xlsx';
    
    
    TD2 = xlsread(td,'lane2',xlRange);
    TD3 = xlsread(td,'lane3',xlRange);
    TD4 = xlsread(td,'lane4',xlRange);
    
    TDns2 = xlsread(tdns,'lane2',xlRange);
    TDns3 = xlsread(tdns,'lane3',xlRange);
    TDns4 = xlsread(tdns,'lane4',xlRange);
    
    CER2 = xlsread(cer,'lane2',xlRange);
    CER3 = xlsread(cer,'lane3',xlRange);
    CER4 = xlsread(cer,'lane4',xlRange);
    
    CERns2 = xlsread(cerns,'lane2',xlRange);
    CERns3 = xlsread(cerns,'lane3',xlRange);
    CERns4 = xlsread(cerns,'lane4',xlRange);

    
    RLMPC2 = xlsread(cermpc,'lane2',xlRange);
    RLMPC3 = xlsread(cermpc,'lane3',xlRange);
    RLMPC4 = xlsread(cermpc,'lane4',xlRange);

end

data = TD2;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('TD2');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = TD3;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('TD3');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = TD4;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('TD4');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

%% 
data = TDns2;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('TDns2');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");


data = TDns3;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('TDns3');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = TDns4;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('TDns4');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

%%
data = CER2;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('CER2');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = CER3;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('CER3');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = CER4;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('CER4');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

%% 
data = CERns2;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('CERns2');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = CERns3;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('CERns3');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = CERns4;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('CERns4');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");


%% 
data = RLMPC2;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('RLMPC2');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = RLMPC3;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('RLMPC3');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

data = RLMPC4;
idx = find(data(:,3) == 1);
deviation = data(idx,6);
speed = data(idx, 1);
disp('RLMPC4');
fprintf("%.2fm\n%.2fm\n%.2fm\n%.2fm/s\n",mean(deviation), max(deviation), min(deviation), mean(speed));
fprintf("average deviation = %.2f \n", mean(deviation))
fprintf("max deviation = %.2f \n", max(deviation))
fprintf("min deviation = %.2f \n", min(deviation))
fprintf("average speed = %.2f \n", mean(speed))
disp("///////////////");

