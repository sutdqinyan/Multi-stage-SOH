function [InputData, Xtrain, YS] = Dataslicing1(InputData, Capacity, Phase, TrainLength, Trainsample1, tau, m)

ReCapacity = [];
Xtrain = [];
for i = Phase(1):Phase(2)
    L = size(InputData{i},2);
    InputData{i}(:,TrainLength+1:end) = [];
end

%% Phase suspace reconstruction
for d = Phase(1):Phase(1)+Trainsample1-1
      N = size(InputData{d},2);
      data = [reconstitution(InputData{d}(1,:), N, m, tau); reconstitution(InputData{d}(2,:), N, m, tau); reconstitution(InputData{d}(3,:), N, m, tau)];
      InputData{d} = data';
      Xtrain = [Xtrain, InputData{d}'];
end
[Xtrain, YS] = mapminmax(Xtrain);