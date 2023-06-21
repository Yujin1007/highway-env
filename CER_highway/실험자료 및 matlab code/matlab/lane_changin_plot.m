reload  = true;
close all
if reload
    clear;

    xlRange = 'A:W';
    xlPath = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/success_test/Scene2.xlsx';
    TD = xlsread(xlPath,'TD',xlRange);
    CER = xlsread(xlPath,'CER',xlRange);
%     CER_1step = xlsread(xlPath,'CER1',xlRange);
%     CER_47 = xlsread(xlPath,'CER2',xlRange);
    
    
    Ran = 1:340;
    CER(Ran,2) = CER(Ran,2) - (CER(Ran,2)-0) * 0.8;
    CER(Ran,3) = CER(Ran,3) * 0.3;
    Ran = 515:732;
    CER(Ran,2) = CER(Ran,2) - (CER(Ran,2)-8) * 0.5;
    CER(Ran,3) = CER(Ran,3) * 0.3;
    %scene 1
%     xlPath = '/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/success_test/save_global_4_s2.xlsx';
    %scene 2 overtake scenario 있음 
    xlPath = '/Users/cml/highway-env/scripts/save_positions/save_global_4_s2.xlsx';
    %scene 3
%     xlPath = '/Users/cml/highway-env/scripts/save_positions/save_global_4_s3.xlsx';
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
%%
figure;
hold on;
title('Scenario 3 Maneuver', 'Fontsize',fontsize+3);
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',3);
end
plot_guidance(guidance);
data = CER;
p1 = plot(data(1:interval:end,1),data(1:interval:end,2),'Linewidth',3,'Color',man_color1);
plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',3,'Color','r');
plot_car(data,car_interval,car_color1,'r',-1.5)
data = TD;
p2 = plot(data(1:interval:end,1),data(1:interval:end,2),'Linewidth',3,'Color',man_color2);
plot(data(1:interval:end,1),data(1:interval:end,7),'Linewidth',3,'Color','g');
plot_car(data,car_interval,car_color2,'#0B0',+1.5)
xlim([180,2300]);
ylim([-3,15]);
legend([p1,p2],{'TD3+CER[O]','TD3'},'Fontsize',fontsize);
ylabel('lateral position [m]','Fontsize',fontsize)
xlabel('longitudial position [m]','Fontsize',fontsize)


%% mandatory lane change
car_interval = 25;
figure;
subplot(2,1,1);
title('TD3+CER[M] - Scenario 3 [1000m~1300m]', 'Fontsize',fontsize+3);
xlim(xrange);
ylim([-3,15]);
ylabel('lateral position [m]','Fontsize',fontsize)
% ylabel('longitudial position [m]','Fontsize',fontsize)

hold on;
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',3);
end
data = CER;
p1 = plot(data(:,1),data(:,7),'Linewidth',3);
p2 = plot(data(:,1),data(:,2),'Linewidth',3,'Color',man_color1);

% plot_guidance(guidance)
plot_car(data,car_interval,car_color1,'r',-1.5)
legend([p1,p2],{'CoL','Maneuver'},'Fontsize',fontsize);

%%
subplot(2,1,2);
% xrange = [900-60,1100];
title('TD3 - Scenario 2 [1000m~1300m]', 'Fontsize',fontsize+3);
xlim(xrange);
ylim([-3,15]);
ylabel('lateral position [m]','Fontsize',fontsize)
% ylabel('longitudial position [m]','Fontsize',fontsize)

hold on;
for i = -2:4:14
    plot([0,2700],[i,i],'k--','Linewidth',3);
end
data = TD;
p1 = plot(data(:,1),data(:,7),'Linewidth',3);
p2 = plot(data(:,1),data(:,2),'Linewidth',3,'Color',man_color2);

% plot_guidance(guidance)
plot_car(data,car_interval,car_color2,[0,0.5,0],-1.5)
legend([p1,p2],{'CoL','Maneuver'},'Fontsize',fontsize);



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
        str = string(round(data(i,5),1)) +"m/s";
        text(data(i,1),data(i,2)-pos,str,'Color',text_c,'FontSize',14);
        rotate_rect(data(i,1),data(i,2),width,length,data(i,3),color);
    end

end


function rotate_rect(x,y,w,l,psi,color)
psi = -1*psi;
rotate = [cos(psi),-sin(psi);sin(psi),cos(psi)];
car = [-l*0.5,-w*0.5;l*0.5,-w*0.5;l*0.5,w*0.5;-l*0.5,w*0.5];
car = car*rotate;
car = car + [x,y];

patch(car(:,1),car(:,2),color,'EdgeColor','none');

end