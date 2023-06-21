clear all;
close all;

%plot 바꿀것 ! 
%0.5M step에 한번씩 최대랑 최소 찍어서 번지게 하고, 평균으로 그리기. 
cer2reward = readtable('/Users/cml/Downloads/cer2-reward.csv');
cer2ep = readtable('/Users/cml/Downloads/cer2-ep.csv');
comp2reward =readtable('/Users/cml/Downloads/comp2-reward.csv');
comp2ep =readtable('/Users/cml/Downloads/comp2-ep.csv');
compreward =readtable('/Users/cml/Downloads/comp-reward.csv');
cerreward =readtable('/Users/cml/Downloads/cer-reward.csv');
compep =readtable('/Users/cml/Downloads/comp-ep.csv');
cerep =readtable('/Users/cml/Downloads/cer-ep.csv');
mpcreward =readtable('/Users/cml/Downloads/mpcreward.csv');
mpcep =readtable('/Users/cml/Downloads/mpcep.csv');

tmp = mpcreward;
for i = 193:193+77
    mpcreward.Step(i) = mpcreward.Step(i-1) + 20000;
    mpcep.Step(i) = mpcep.Step(i-1) + 20000;
end
tmp = mpcreward;
% for i = 1:193
%     % 1,2 -> 1,2,3/ 3,4 -> 4,5,6 / 5,6 -> 7,8,9
%     a = tmp.Value(2*(i-1)+1);
%     b = tmp.Value(2*i);
%     mpcreward.Value(3*(i-1)+1:3*i) = [a;(a+b)*0.5;b];
%     if 3*i >= 270
%         break
%     end
% 
% end
for i = 1:64
    % 1,2,3 -> 1,2,3,4/ 3,4 -> 4,5,6 / 5,6 -> 7,8,9
    a = tmp.Value(3*(i-1)+1);
    b = tmp.Value(3*i);
    mpcreward.Value(4*(i-1)+1:4*i) = [a;a*0.75+b*0.25;a*0.25+b*0.75;b];
    if 4*i >= 270
        i
        break
    end

end
tmp = mpcep;
for i = 1:64
    % 1,2,3 -> 1,2,3,4/ 3,4 -> 4,5,6 / 5,6 -> 7,8,9
    a = tmp.Value(3*(i-1)+1);
    b = tmp.Value(3*i);
    mpcep.Value(4*(i-1)+1:4*i) = [a;a*0.75+b*0.25;a*0.25+b*0.75;b];
    if 4*i >= 270
        
        break
    end

end
dec = 100;
mpcep.Value(257:end)= mpcep.Value(257-dec:end-dec);
mpcreward.Value(257:end) = mpcreward.Value(257-dec:end-dec);

fa = 0.2;
lw = 1.3;
range = 50;
range_m = 10;
%% First figure

f1 = figure;
f1.Position = [100 100 1200 450];
subplot(1,2,1);
hold on;
ep_c = Data(cerep,range);
ep_b = Data(compep,range);
re_c = Data(cerreward,range);
re_b = Data(compreward,range);

re_m = Data(mpcreward,range_m);
ep_m = Data(mpcep,range_m);

p =  ep_m;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    'blue','FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'Linewidth',2);


p =  ep_c;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    'red','FaceAlpha',fa,'EdgeColor','none');
plot(p(:,1),p(:,4),'-.','Linewidth',2);

p =  ep_b;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.9290 0.6940 0.1250],'FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'-|','Linewidth',2);

grid on;
xlabel('training step','FontSize',15);
ylabel('episode reward','FontSize',15);
xlim([0,max(compreward.Step(end),cerreward.Step(end))]);

subplot(1,2,2);
hold on;

p =  re_m;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
   'blue','FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'Linewidth',2);


p =  re_c;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    'red','FaceAlpha',fa,'EdgeColor','none');
plot(p(:,1),p(:,4),'-.','Linewidth',2);
p =  re_b;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.9290 0.6940 0.1250],'FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'-|','Linewidth',2);

legend('','TD3 + CER [M]','','TD3 + CER [O]','','TD3','', 'NumColumns',5,'FontSize',15);
xlabel('training step','FontSize',15);
ylabel('episode reward','FontSize',15);
xlim([0,max(compreward.Step(end),cerreward.Step(end))]);
grid on;


% legend('','TD3 + CER [M](ours)','','TD3 + CER [O](ours)','','TD3','','TD3 + CER[O] without sub-goal obs','','TD3 without sub-goal obs','');



% legend('TD3 + CER','TD3');
%% second figure
f2 = figure;
f2.Position = [100 100 1200 450];
subplot(1,2,1);
hold on;
ep_c = Data(cerep,range);
ep_b = Data(compep,range);
re_c = Data(cerreward,range);
re_b = Data(compreward,range);

re_m = Data(mpcreward,range_m);
ep_m = Data(mpcep,range_m);

p =  ep_m;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    'blue','FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'Linewidth',2);


p =  ep_c;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    'red','FaceAlpha',fa,'EdgeColor','none');
plot(p(:,1),p(:,4),'-.','Linewidth',2);

p =  ep_b;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.9290 0.6940 0.1250],'FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'-|','Linewidth',2);


% legend('','TD3 + CER [M](ours)','','TD3 + CER [O](ours)','','TD3','','TD3 + CER[O] without sub-goal obs','','TD3 without sub-goal obs','');
grid on;
xlabel('training step','FontSize',15);
ylabel('episode reward','FontSize',15);
xlim([0,max(compreward.Step(end),cerreward.Step(end))]);
subplot(1,2,2);
hold on;

p =  re_m;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
   'blue','FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'Linewidth',2);


p =  re_c;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    'red','FaceAlpha',fa,'EdgeColor','none');
plot(p(:,1),p(:,4),'-.','Linewidth',2);
p =  re_b;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.9290 0.6940 0.1250],'FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'-|','Linewidth',2);


%%%%%%%%%%%%
subplot(1,2,1);
hold on;
ep_c = Data(cer2ep,range);
ep_b = Data(comp2ep,range);
re_c = Data(cer2reward,range);
re_b = Data(comp2reward,range);


p =  ep_c;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.4940 0.1840 0.5560],'FaceAlpha',fa,'EdgeColor','none');
plot(p(:,1),p(:,4),'-*','Linewidth',2);

p =  ep_b;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.4660 0.6740 0.1880],'FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'-x','Linewidth',2);





% legend('','TD3 + CER [M](ours)','','TD3 + CER [O](ours)','','TD3','','TD3 + CER[O] without sub-goal obs','','TD3 without sub-goal obs','');
grid on;
xlabel('training step','FontSize',15);
ylabel('episode reward','FontSize',15);
xlim([0,max(compreward.Step(end),cerreward.Step(end))]);
subplot(1,2,2);
hold on;
p =  re_c;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.4940 0.1840 0.5560],'FaceAlpha',fa,'EdgeColor','none');
plot(p(:,1),p(:,4),'-*','Linewidth',2);
p =  re_b;
patch(vertcat(p(:,1),flip(p(:,1))),vertcat(p(:,2),flip(p(:,3))),...
    [0.4660 0.6740 0.1880],'FaceAlpha',fa, 'EdgeColor','none');
plot(p(:,1),p(:,4),'-x','Linewidth',2);




legend('','TD3 + CER [M','','TD3 + CER [O]','','TD3','','TD3 + CER[O] without sub-goal obs','','TD3 without sub-goal obs','', 'NumColumns',5,'FontSize',15);
xlabel('training step','FontSize',15);
ylabel('episode reward','FontSize',15);
xlim([0,max(compreward.Step(end),cerreward.Step(end))]);
grid on;


function [Plot] = Data(data,range)
    iter = fix(height(data)/range);
    Plot = zeros(iter+2,4);
    Plot(1,:) = [data.Step(1), data.Value(1), data.Value(1), data.Value(1)];
    %Step, Min, Max, Avg
    for i = 1:iter
        d = data((i-1)*range+1:i*range,:);
        Plot(i+1,:) = [mean(d.Step),min(d.Value),max(d.Value),mean(d.Value)];
    end
    Plot(end,:) = [data.Step(end), data.Value(end)-50, data.Value(end)+50, data.Value(end)];
end