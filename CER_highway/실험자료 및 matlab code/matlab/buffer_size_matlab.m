% 필요한것 : tag, index 
clear;

buffer_size = 2000;
batch_size = 3;
iter = 1000;
Record = zeros(iter, 2);

% [buffer, cnt0, cnt1] = Initbuffer(buffer_size);
[buffer, cnt0, cnt1] = Initbuffer(20); %20개만 initialize
lc_record = zeros(cnt0,1);
lk_record = zeros(cnt1,1);


for i = 1:iter
    batch_idx = randsample(height(buffer), batch_size);
    batch = buffer(batch_idx,:);
    lc_batch = batch(find(batch(:,1) == 0),:);
    lk_batch = batch(find(batch(:,1) == 1),:);
    Record(i,:) = [height(lk_batch), height(lc_batch)];
    
    lc_record(lc_batch(:,2)) = lc_record(lc_batch(:,2)) + 1;
    lk_record(lk_batch(:,2)) = lk_record(lk_batch(:,2)) + 1;
%     decision = randDi(i,iter);
    decision = randD();
    [buffer, cnt0, cnt1,lc_record,lk_record] = Addbuffer(buffer, cnt0, cnt1,lc_record,lk_record,buffer_size,decision);
end
fprintf('stored = %.0f, replay/LCsampe = %.2f\n',length(find(buffer(:,1) == 0)),mean(lc_record));
fprintf('stored = %.0f, replay/LKsampe = %.2f\n',length(find(buffer(:,1) == 1)),mean(lk_record));
fprintf('total LC update : %.0f\n', sum(lc_record));
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

function [buffer, cnt0, cnt1] = Initbuffer(buffer_size)
    buffer = zeros(buffer_size,2);
    cnt0 = 0;
    cnt1 = 0;
    for i = 1:buffer_size
%         decision = randDi(0,1);
        decision = randD();
        if decision == 1
            cnt1 = cnt1 + 1;
            buffer(i,:) = [decision, cnt1];
        else
            cnt0 = cnt0 + 1;
            buffer(i,:) = [decision, cnt0];
        end
    end
end

function [buffer, cnt0, cnt1,lc_record,lk_record] = Addbuffer(buffer, cnt0, cnt1,lc_record,lk_record,buffer_size,decision)


    if decision == 1 
        cnt1 = cnt1 + 1;
        lk_record(end+1) = 0;
        if height(buffer) < buffer_size
            buffer(end+1,:) = [decision, cnt1];
        else
            buffer(1:end-1, :) = buffer(2:end, :);
            buffer(end,:) = [decision, cnt1];
        end
        
    else
        cnt0 = cnt0 + 1;
        lc_record(end+1) = 0;
        if height(buffer) < buffer_size
            buffer(end+1,:) = [decision, cnt0];
        else
            buffer(1:end-1, :) = buffer(2:end, :);
            buffer(end,:) = [decision, cnt0];
        end
        
        
    end

end
