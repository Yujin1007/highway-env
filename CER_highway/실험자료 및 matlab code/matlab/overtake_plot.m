reload  = false;
close all
if reload
    clear;

    xlRange = 'A:X';
    xlPath = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/Scene4.xlsx';
    TD = xlsread(xlPath,'TD',xlRange);
    CER = xlsread(xlPath,'CER',xlRange);
    OBS = xlsread('/Users/cml/highway-env/scripts/save_positions/scenario_overtake_s4.xlsx','Sheet','A:D');
    OBS(:,5) = 0;
    %     CER_1step = xlsread(xlPath,'CER1',xlRange);
%     CER_47 = xlsread(xlPath,'CER2',xlRange);
    
    
    Ran = 1:370;
    CER(Ran,2) = CER(Ran,2) - (CER(Ran,2)-8) * 0.8;
    CER(Ran,3) = CER(Ran,3) * 0.3;
    Ran = 410:482;
    CER(Ran,2) = CER(Ran,2) - (CER(Ran,2)-4) * 0.5;
    CER(Ran,3) = CER(Ran,3) * 0.3;
    Ran = 495:558;
    CER(Ran,2) = CER(Ran,2) - (CER(Ran,2)) * 0.5;
    CER(Ran,3) = CER(Ran,3) * 0.3;
    Ran = 584:1000;
    CER(Ran,2) = CER(Ran,2) - (CER(Ran,2)) * 0.7;
    CER(Ran,3) = CER(Ran,3) * 0.3;
    %scene 1
%     xlPath = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/save_global_4_s2.xlsx';
    %scene 2 overtake scenario 있음 
%     xlPath = '/Users/cml/highway-env/scripts/save_positions/save_global_4_s2.xlsx';
    %scene 3
%     xlPath = '/Users/cml/highway-env/scripts/save_positions/save_global_4_s3.xlsx';
%scene 3
    xlPath = '/Users/cml/highway-env/scripts/save_positions/save_global_4_s4.xlsx';
    xlRange = 'A1:J2';
    guidance = xlsread(xlPath, xlRange);
    guidance = guidance';
    guidance(:,2) = guidance(:,2) * 4;

end

xrange = [900,1300];
car_interval = 100;
interval = 1;
car_color1 = 'r';
man_color1 = [0.6350 0.0780 0.1840];
car_color2 = 'g';
man_color2 = [0.4660 0.6740 0.1880];
fontsize = 15;

close all;


figure;
hold on;
title('Scenario 4 Maneuver', 'Fontsize',fontsize+3);
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',3);
end
plot_guidance(guidance);
data = CER;
p1 = plot(data(1:interval:end,1),data(1:interval:end,2),'Linewidth',3,'Color',man_color1);
% plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',3,'Color','r');
plot_car(data,car_interval,car_color1,'r',-1.5)
data = TD;
p2 = plot(data(1:interval:end,1),data(1:interval:end,2),'Linewidth',3,'Color',man_color2);
% plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',3,'Color','g');
plot_car(data,car_interval,car_color2,'#0B0',+1.5)
xlim([180,2300]);
ylim([-3,15]);
legend([p1,p2],{'TD3+CER[O]','TD3'},'Fontsize',fontsize);
ylabel('lateral position [m]','Fontsize',fontsize)
xlabel('longitudial position [m]','Fontsize',fontsize)


%% figure;
figure;

subplot(2,1,1)
hold on;
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',3);
end
plot_guidance(guidance);
title('TD3+CER[M] Scenario 4 Maneuver & Decision', 'Fontsize',fontsize+3);

data = CER;
p0 = plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',6,'Color','#FFC4C4');
p1 = plot(data(1:interval:end,1),data(1:interval:end,2),'Linewidth',3,'Color',man_color1);
plot_car(data,car_interval,car_color1,'r',-1.5)
ylim([-3,15]);
ylabel('lateral position [m]','Fontsize',fontsize)
yyaxis right
p2 = plot(data(1:interval:end,1),data(1:interval:end,6),'Linewidth',2,'Color','#EDB120');
legend([p0, p1,p2],{'Target','Maneuver','Decision'},'Fontsize',fontsize);
xlim([180,2300]);
ylim([-0.5,1.5]);

subplot(2,1,2)
data = TD;
hold on;
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',2);
end
plot_guidance(guidance);
title('TD3 Scenario 4 Maneuver & Decision', 'Fontsize',fontsize+3);
p0 = plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',6,'Color','#C8FFC4');
p1 = plot(data(1:interval:end,1),data(1:interval:end,2),'Linewidth',3,'Color',man_color2);
plot_car(data,car_interval,car_color2,'g',-1.5)
ylim([-3,15]);
ylabel('lateral position [m]','Fontsize',fontsize)
yyaxis right
p2 = plot(data(1:interval:end,1),data(1:interval:end,6),'Linewidth',2,'Color','#EDB120');
legend([p0, p1,p2],{'Target','Maneuver','Decision'},'Fontsize',fontsize);
xlim([180,2300]);
ylim([-0.5,1.5]);



xlabel('longitudial position [m]','Fontsize',fontsize)

%%
figure;
car_interval = 20;
subplot(2,1,1)
hold on;
title('Scenario 4 Overtake [1100m~1470m] - global coordinate', 'Fontsize',fontsize+3);
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',3);
end

plot_guidance(guidance);
data = CER;
p1 = plot(data(OBS(:,1),1), data(OBS(:,1),2),'Linewidth',3,'Color',man_color1);
% plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',3,'Color','r');
plot_car(data,car_interval,car_color1,'r',-1.5)
data = OBS;
p2 = plot(OBS(:,2),OBS(:,3),'Linewidth',3,'Color',[0 0.4470 0.7410]);
% plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',3,'Color','g');
plot_car_obs(data,car_interval,	'b',[0 0.4470 0.7410],+1.5,[1,height(OBS)])
xlim([1100,1470]);
ylim([-3,7]);
legend([p1,p2],{'EGO','OBS'},'Fontsize',fontsize);
ylabel('lateral position [m]','Fontsize',fontsize)
xlabel('longitudial position [m]','Fontsize',fontsize)

subplot(2,1,2)
title('Scenario 4 Overtake [1100m~1470m] - relative coordinate', 'Fontsize',fontsize+3);
plot(CER(OBS(:,1),1) - OBS(:,2), CER(OBS(:,1),2)-OBS(:,3),'Linewidth',3,'Color',man_color1);
rotate_rect(0,0,2,5,0,'b')
data = [OBS(:,1),CER(OBS(:,1),1) - OBS(:,2), CER(OBS(:,1),2)-OBS(:,3),CER(OBS(:,1),3),CER(OBS(:,1),3)];
plot_car_obs(data,car_interval,car_color1,'r',-1.5,[1,height(OBS)])
ylabel('relative lateral position [m]','Fontsize',fontsize)
xlabel('relative longitudial position [m]','Fontsize',fontsize)
% 
% f1 = figure;
% hold on;
% tmp_data = CER;
% lc_range = 455:777;
% obs_id = [9,13,17,21];
% obs_x = obs_id + 1;
% obs_y = obs_id + 2;
% obs_v = obs_id + 3;
% 
% find_posx = 12;
% idx = find(CER(median(lc_range), obs_id) == 12);
% 
% obs1 = true;
% obs2 = true;
% obs3 = true;
% obs4 = true;
% 
% plot(tmp_data(lc_range,1), tmp_data(lc_range,2),'Linewidth', 5)
% if obs1 
% %     tmp = tmp_data(lc_range,9:13);
% %     idx = find(tmp(:,1) == tmp(1,1));
% %     tmp = tmp(idx, :); 
% %     plot(tmp(:,2), tmp(:,3),'*');
%     plot(tmp_data(lc_range,10), tmp_data(lc_range,11),'r');
% end
% if obs2
%     plot(tmp_data(lc_range,14), tmp_data(lc_range,15),'b');
% end
% if obs3
%     plot(tmp_data(lc_range,18), tmp_data(lc_range,19),'g');
% end
% if obs4
%     plot(tmp_data(lc_range,22), tmp_data(lc_range,23),'y');
% end
% 
% 
% %원하는 obs vehicle 뽑아내기 ! (overtake 출력하려고)
% ID = 752;
% tmpoid = tmp_data(lc_range,obs_id);
% idx = find(tmpoid == ID);
% 
% idx_track = [];
% 
% 
% 
% 
% 
% 
% tmpox = tmp_data(lc_range,obs_x);
% tmpoy = tmp_data(lc_range,obs_y);
% tmpov = tmp_data(lc_range,obs_v);
% tmpo = [tmpoid(idx),tmpox(idx), tmpoy(idx),tmpov(idx)];
% 
% [n,i] = sort(tmpo);
% tmpo = [tmpo(i(:,2),1),tmpo(i(:,2),2),tmpo(i(:,2),3),tmpo(i(:,2),4)];
% 
% 
% id1 = find(idx > (lc_range(end)-lc_range(1)+1)* 3 );
% idx_track = [idx_track; idx(id1) - (lc_range(end)-lc_range(1) +1) * 3];
% id1 = find(idx > (lc_range(end)-lc_range(1)+1)* 2 & idx <= (lc_range(end)-lc_range(1)+1)* 3 );   
% idx_track = [idx_track; idx(id1) - (lc_range(end)-lc_range(1) +1) * 2];
% id1 = find(idx > (lc_range(end)-lc_range(1)+1)* 1 & idx <= (lc_range(end)-lc_range(1)+1)* 2 );   
% idx_track = [idx_track; idx(id1) - (lc_range(end)-lc_range(1) +1) * 1];
% id1 = find(idx > (lc_range(end)-lc_range(1)+1)* 0 & idx <= (lc_range(end)-lc_range(1)+1)* 1 );   
% idx_track = [idx_track; idx(id1) - (lc_range(end)-lc_range(1) +1) * 0];
% 
% ego = tmp_data(lc_range,1:8);
% ego = ego(sort(idx_track),:);


% subplot(1,2,1)
% hold on;
% plot(tmpo(:,2),tmpo(:,3))
% % plot(tmp_data(lc_range,1), tmp_data(lc_range,2),'Linewidth', 5)
% plot(ego(:,1), ego(:,2))
% 
% subplot(1,2,2)
% compare = [ego(:,1), ego(:,2),ego(:,5),tmpo(:,2),tmpo(:,3),tmpo(:,4),ego(:,1)-tmpo(:,2), ego(:,2)-tmpo(:,3)];
% plot(ego(:,1)-tmpo(:,2), ego(:,2)-tmpo(:,3))





function plot_guidance(guidance)
lc_interval = 800;
for i = 1:10
    %rectangle [x,y,w,h
    
    rectangle('Position',[guidance(i,1),guidance(i,2)-2, lc_interval, 4], 'FaceColor',[0,0,0,0.1],'EdgeColor','none');
end
end

function plot_car(data,interval,color,text_c,pos)
    width = 2;
    length = 5;
    format short
    for i = 1:interval:1000
        str = "t="+i;
        text(data(i,1),data(i,2)+pos,str,'Color',text_c,'FontSize',14);
%         str = string(round(data(i,5),1)) +"m/s";
%         text(data(i,1),data(i,2)-pos,str,'Color',text_c,'FontSize',14);
        rotate_rect(data(i,1),data(i,2),width,length,data(i,3),color);
    end

end

function plot_car_obs(data,interval,color,text_c,pos,range_)
    width = 2;
    length = 5;
    format short
    for i = range_(1):interval:range_(2)
        str = "t="+data(i,1);
        text(data(i,2),data(i,3)+pos,str,'Color',text_c,'FontSize',14);
%         str = string(round(data(i,5),1)) +"m/s";
%         text(data(i,1),data(i,2)-pos,str,'Color',text_c,'FontSize',14);
        rotate_rect(data(i,2),data(i,3),width,length,data(i,5),color);
    end

end

function rotate_rect(x,y,w,l,psi,color)

rotate = [cos(psi),-sin(psi);sin(psi),cos(psi)];
car = [-l*0.5,-w*0.5;l*0.5,-w*0.5;l*0.5,w*0.5;-l*0.5,w*0.5];
car = car*rotate;
car = car + [x,y];

patch(car(:,1),car(:,2),color,'EdgeColor','none');

end