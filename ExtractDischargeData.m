function [Data, Capacity] = ExtractDischargeData(BatteryName)
count = 0; Data = []; Capacity = [];
for c = 1:size(BatteryName.cycle, 2)
    data = [];
    if strcmp(BatteryName.cycle(c).type,'discharge')
       count = count + 1;
       data = [data; BatteryName.cycle(c).data.Voltage_measured];
       data = [data; BatteryName.cycle(c).data.Current_measured];
       data = [data; BatteryName.cycle(c).data.Temperature_measured];
       Data{count} = data;
       Capacity(count) = BatteryName.cycle(c).data.Capacity;
    end
end