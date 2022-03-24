function OutputData = DataRemoving(InputData,Number)

for c = 1:size(InputData,2)
    [minu,index] = min(InputData{c}(1,:));
    InputData{c}(:,index+1:end) = [];
    InputData{c}(:,1:Number) = [];
end
OutputData = InputData;