% 필요한것 : tag, index 
clear;

% buffer_size1 = 1e6;
% buffer_size0 = 5e5;
% 
% batch_size = 256;
buffer_size1 = 2500;
buffer_size0 = 2500;

batch_size = 3;
iter = 10000;
Record = zeros(iter, 2);

% [buffer0,buffer1] = Initbuffer(buffer_size0, buffer_size1);
[buffer0,buffer1] = Initbuffer(10, 10);
cnt0 = 10;
cnt1 = 10;

lc_record = zeros(cnt0,1);
lk_record = zeros(cnt1,1);

% cnt0 = buffer_size0;
% cnt1 = buffer_size1;


for i = 1:iter
%     decision = randDi(i,iter);
    decision = randD();
    if decision == 1
        batch_idx = randsample(height(buffer1), batch_size);
        batch = buffer1(batch_idx,:);
        lk_record(batch(:,2)) = lk_record(batch(:,2)) + 1;
        Record(i,:) = [batch_size, 0];
        [buffer1, cnt1, lk_record] = Addbuffer(buffer1, cnt1,lk_record,decision,buffer_size1);
    else
        batch_idx = randsample(height(buffer0), batch_size);
        batch = buffer0(batch_idx,:);
        lc_record(batch(:,2)) = lc_record(batch(:,2)) + 1;
        Record(i,:) = [0, batch_size];
        [buffer0, cnt0, lc_record] = Addbuffer(buffer0, cnt0,lc_record,decision,buffer_size0);
    end
    
    
    
end

fprintf('stored = %.0f, replay/LCsampe = %.2f\n',height(buffer0), mean(lc_record));
fprintf('stored = %.0f, replay/LKsampe = %.2f\n',height(buffer1), mean(lk_record));
fprintf('total LC update : %.0f\n\n', sum(lc_record));


function decision = randD()
    % keep : change = 20 : 1 
    % decision -> keep = 1, change = 0 
    x = randi(210);
    
    if x <= 200
        decision = 1;
    else
        decision = 0;
    end
end

function decision = randDi(i,iter)
    % keep : change = 20 : 1 
    % decision -> keep = 1, change = 0 
    x = randi(100);
    
    if x <= 50 + i/iter*49
        decision = 1;
    else
        decision = 0;
    end
end

function [buffer0,buffer1] = Initbuffer(buffer_size0, buffer_size1)
    buffer0 = zeros(buffer_size0,2);
    buffer1 = ones(buffer_size1,2);
    
    buffer0(:,2) = linspace(1,buffer_size0,buffer_size0);
    buffer1(:,2) = linspace(1,buffer_size1,buffer_size1);
    
end

function [buffer, cnt,record] = Addbuffer(buffer, cnt,record,decision, buffer_size)
    cnt = cnt + 1;
    record(end+1) = 0;
    if height(buffer) < buffer_size
        buffer(end+1,:) = [decision,cnt];
    else
        buffer(1:end-1, :) = buffer(2:end, :);
        buffer(end,:) = [decision, cnt];
    end
end
