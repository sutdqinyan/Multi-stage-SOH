clc; clear; close all

load('data\Data_bettery.mat')
Phase1 = [1,30]; Phase2 = [31,108]; Phase3 = [109,166];

%% Extract the discharging cycles
figure
plot(Capacity_No5)
hold on
plot(Capacity_No6)
hold on
plot(Capacity_No7)
legend('B5','B6','B7')
xlabel("No. of cycles")
ylabel("Capacity (Amp-hr)")
xlim([0, size(Capacity_No7,2)])

figure
for i = 31:size(DATA_No5,2)
    plot(DATA_No5{i}(1,:))
    hold on
end
xlim([0,355])
xlabel('Sampling')
ylabel('Voltage (V)')


