close all;
%% aggressive -> tolerance
%% passive -> aggressive
%% data load 
xlRange = 'A2:B1001';
xlPath = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/driving style/CER4.xlsx';
veloPath = '/Users/cml/highway-env/scripts/MPC/RL_MPC_buffer_2/RLMPC.xlsx';
mpcPath = '/Users/cml/highway-env/scripts/MPC/MPC_original.xlsx';
mpcPath2 = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/MPC/CER_MPC.xlsx';

ORIGINAL = 'TrialM68_7_4';
VELOCITY = 'Trialaction5';
FRONT = 'front_dist';
MPC = 'Trial01';
CERMPC = 'CER_M';
TD3 = 'TD3';

original = xlsread(xlPath,ORIGINAL,xlRange);
velocity = xlsread(veloPath,VELOCITY,'A2:E51');
mpc = xlsread(mpcPath,MPC,'A2:K1001');
cermpc = xlsread(mpcPath2,CERMPC,'A2:K1001');
td3 = xlsread(mpcPath2,TD3,'A2:K1001');
% passive = xlsread(xlPath,PASSIVE,xlRange);
% aggressive = xlsread(xlPath,AGGRESSIVE,xlRange);
front = xlsread(xlPath,FRONT,'A1:ALL3');
front = front';
idx = find(front(:) > 100);
front(idx) = 100;


p = original;
p(2:end,3) = atan2(p(2:end,2)-p(1:end-1,2), p(2:end,1)-p(1:end-1,1));
original = p;
p = passive;
p(2:end,3) = atan2(p(2:end,2)-p(1:end-1,2), p(2:end,1)-p(1:end-1,1));
passive = p;
p = aggressive;
p(2:end,3) = atan2(p(2:end,2)-p(1:end-1,2), p(2:end,1)-p(1:end-1,1));
aggressive = p;

xlRange = 'E2:X1001';
original(:,4:23) = xlsread(xlPath,ORIGINAL,xlRange);
passive(:,4:23) = xlsread(xlPath,PASSIVE,xlRange);
aggressive(:,4:23) = xlsread(xlPath,AGGRESSIVE,xlRange);
% 인근 x 좌표 : 9,13,17,21


xlRange = 'C2:D1001';
original(:,24:25) = xlsread(xlPath,ORIGINAL,xlRange);

interval = 20;
lc_interval = 800;
xlPath = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/save_global_4.xlsx';
% xlRange = 'A1:J2';
% guidance = xlsread(xlPath, xlRange);
xlRange = 'Y2:Z26';
guidance = xlsread(mpcPath2,CERMPC,xlRange);

% guidance = guidance';
% guidance(:,2) = guidance(:,2) * 4;


designated_obey_o = designated_lane(original(:,1:2),guidance);
designated_obey_p = designated_lane(passive(:,1:2),guidance);
designated_obey_a = designated_lane(aggressive(:,1:2),guidance);

error = deviation(original)
%% plot First

subplot(2,1,1);
hold on;
grid on;
% for i = 1:10
%     %rectangle [x,y,w,h
%     rectangle('Position',[guidance(i,1),guidance(i,2)-2, lc_interval, 4], 'FaceColor',[0,0,0,0.1],'EdgeColor','none');
% end
lw = 2;
xlim([0,2700]);
ylim([-4,16]);
ylabel('lateral position (m)','Fontsize',12);
plot(mpc(1:interval:1000,1),mpc(1:interval:1000,2),'-','Linewidth',lw)
plot(original(1:interval:1000,1),original(1:interval:1000,2),'-','Linewidth',lw)
plot(cermpc(1:interval:1000,1),cermpc(1:interval:1000,2),'-','Linewidth',lw)
plot(td3(1:interval:1000,1),td3(1:interval:1000,2),'-','Linewidth',lw)
% plot(aggressive(1:interval:1000,1),aggressive(1:interval:1000,2),'d-','Linewidth',1.5)
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',0.5);
end


% legend('RL','MPC')
%%
% subplot(3,1,2);
% hold on;
% grid on;
% xlim([0,2700]);
% plot(original(1:interval:1000,1),original(1:interval:1000,24),'*-','Linewidth',1.5)
% % plot(passive(1:interval:1000,1),front(1:interval:1000,3),'o-','Linewidth',1.5)
% % plot(aggressive(1:interval:1000,1),front(1:interval:1000,1),'d-','Linewidth',1.5)
% ylabel('Steering Angle(rad)','Fontsize',12);
% % plot(original(1:interval:1000,1),original(1:interval:1000,4),'Linewidth',3)
% % plot(passive(1:interval:1000,1),passive(1:interval:1000,4),'Linewidth',3)
% % plot(aggressive(1:interval:1000,1),aggressive(1:interval:1000,4),'Linewidth',3)

subplot(2,1,2);
hold on;
grid on;
xlim([0,2700]);
ylim([15,35]);
plot(mpc(1:interval:1000,1),mpc(1:interval:1000,5),'-','Linewidth',lw)
plot(velocity(1:50,1),velocity(1:50,5),'-','Linewidth',lw)
plot(cermpc(1:interval:1000,1),cermpc(1:interval:1000,5),'-','Linewidth',lw)
plot(td3(1:interval:1000,1),td3(1:interval:1000,5),'-','Linewidth',lw)
% plot(aggressive(1:interval:1000,1),aggressive(1:interval:1000,4),'d-','Linewidth',1.5)
ylabel('Velocity (m/s)','Fontsize',12);
plot([0,2700],[30,30],'k--','Linewidth',1.5)
% legend('Avg.: 25.2 m/s ','Avg.: 25,1 m/s','Avg.: 22.45 m/s','')

legend('MPC','TD3+CER[M]','TD3+CER[O]','TD3', 'FontSize',15,'NumColumns',4)

%% plot 3 independently
% 
% figure;
% subplot(3,1,1);
% hold on;
% % grid on;
% xlim([0,2700]);
% ylim([-4,16]);
% plot(original(1:interval:1000,1),original(1:interval:1000,2),'*-','Linewidth',1.5,'Color',[0 0.4470 0.7410])
% for i = -2:4:14
%     plot([0,2700],[i,i],'k--','Linewidth',0.5);
% end
% subplot(3,1,2);
% hold on;
% % grid on;
% xlim([0,2700]);
% ylim([-4,16]);
% plot(passive(1:interval:1000,1),passive(1:interval:1000,2),'o-','Linewidth',1.5,'Color',[0.8500 0.3250 0.0980])
% for i = -2:4:14
%     plot([0,2700],[i,i],'k--','Linewidth',0.5);
% end
% subplot(3,1,3);
% hold on;
% % grid on;
% xlim([0,2700]);
% ylim([-4,16]);
% plot(aggressive(1:interval:1000,1),aggressive(1:interval:1000,2),'d-','Linewidth',1.5,'Color',[0.9290 0.6940 0.1250])
% for i = -2:4:14
%     plot([0,2700],[i,i],'k--','Linewidth',0.5);
% end


% figure;
% hold on;
% plot_car(original,interval,	[0 0.4470 0.7410]);
% plot_car(passive,interval,	[0.8500 0.3250 0.0980]);
% plot_car(aggressive,interval,[0.9290 0.6940 0.1250]);


% axis equal;

function m_chk = designated_lane(data,guidance)
    chk = zeros(1000,1);
    for i = 1:1000
        if data(i,2) < 2
            data(i,2) = 0;
        elseif data(i,2) >=2 && data(i,2) < 6
            data(i,2) = 4;
        elseif data(i,2) >=6 && data(i,2) < 10
            data(i,2) = 8;
        elseif data(i,2) >=10
            data(i,2) = 12;
        end
    end
    j = 1;
    for i = 1:1000
        if data(i,1) > guidance(j+1,1)
            j = j+1;
        end
        if data(i,2) == guidance(j,2)
            chk(i) = 1;
        else
            chk(i) = 0;
        end
    end
    m_chk = mean(chk);
end

function error = deviation(data)
    idx = find(data(:,5) == 1);
    data = data(idx,:);
    l = length(idx);
    
    
    dev = sqrt((data(:,6) - data(:,2)).^2);
    error = mean(dev);
end


function plot_car(data,interval,color)
    width = 2;
    length = 5;
    for i = 1:interval:1000
        rotate_rect(data(i,1),data(i,2),width,length,data(i,3),color);
    end

end

function rotate_rect(x,y,w,l,psi,color)
rotate = [cos(psi),-sin(psi);sin(psi),cos(psi)];
car = [-l*0.5,-w*0.5;l*0.5,-w*0.5;l*0.5,w*0.5;-l*0.5,w*0.5];
car = car*rotate;
car = car + [x,y];

patch(car(:,1),car(:,2),color,'EdgeColor','none');

end

