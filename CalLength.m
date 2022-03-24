function [TrainLength, Trainsample]  = CalLength(Inputdata, Phaselength, Ratio)

TrainLength = 0;
for i = Phaselength(1):Phaselength(2)
    if i == Phaselength(1)
       TrainLength = size(Inputdata{i},2);
    elseif TrainLength > size(Inputdata{i},2)
       TrainLength = size(Inputdata{i},2);
    end
end
Trainsample = floor((Phaselength(2)-Phaselength(1))*Ratio);