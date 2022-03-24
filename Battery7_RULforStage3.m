clc; clear; close all
tic
load('data\Data_bettery.mat')
%% Extract the discharging cycles from Battery 5 and 6
DATA_B5 = DATA_No7; Capacity_B5 = Capacity_No7;
Capacity_B5(43) = [];
Phase1 = [1,30]; Phase2 = [31,108]; Phase3 = [109,167];

%% Calculate the minimum length of each phase (here is three)
[TrainLength, Trainsample] = CalLength(DATA_B5, Phase3, 0.7);

%% Obtain the non-staionary source from training data
tau = 5; m = 3; s = 4;
[TrData_B5, Xtrain, YS] = Dataslicing1(DATA_B5, Capacity_B5, Phase3, TrainLength, Trainsample, tau, m);
[est_Ps, est_Pn, est_As, est_An, ssa_results] = ssa(Xtrain, s, 'reps', 20, 'equal_epochs', Trainsample, 'random_seed', 12345);
Ts = est_Ps * Xtrain; Tn = est_Pn * Xtrain;
for j = 1:9-s
    Tn(j,:) = smoothdata(Tn(j,:),'gaussian',7);
end
[Tn, YSnew] = mapminmax(Tn);

%% Arranging the training dataset
Xtrain = [];
for d = Phase3(1):Phase3(1)+Trainsample-1
       xtrain = [];
       N = size(TrData_B5{d},1);
       xtrain = mapminmax('apply', TrData_B5{d}', YS);
       xtrain = est_Pn*xtrain;
       for j = 1:9-s
           xtrain(j,:) = smoothdata(xtrain(j,:),'gaussian',7);
       end
       xtrain = mapminmax('apply', xtrain, YSnew);       
       xtrain(:,231:end) = [];
       N = size(xtrain,2);
       cycleID = 1:N;
       Capacity = Capacity_B5(d)*ones(N,1);
       TrData_B5{d} = [(d-Phase3(1))*ones(N,1) cycleID' xtrain' Capacity];
       Xtrain = [Xtrain; TrData_B5{d}];
end

%% Arranging the testing dataset
Xtest = []; TT = [];
for d = Phase3(1)+Trainsample:Phase3(2)
    xtest = [];
    N = size(DATA_B5{d},2);
    data = DATA_B5{d}(:,1:TrainLength);
    N = size(data, 2);
    data = [reconstitution(data(1,:), N, m, tau); reconstitution(data(2,:), N, m, tau); reconstitution(data(3,:), N, m, tau)];
    data = mapminmax('apply', data, YS);
    xtest = est_Pn * data;
    for j = 1:9-s
        xtest(j,:) = smoothdata(xtest(j,:),'gaussian',7);
    end
    xtest = mapminmax('apply', xtest, YSnew);    
    xtest(:,231:end) = [];
    cycleID = 1:size(xtest,2);
    Xtest{d-Trainsample-Phase3(1)+1} = [(d-Trainsample-Phase3(1))*ones(size(xtest,2),1) cycleID' xtest'];
    TT = [TT xtest];
end

%% Arranging the data for the following LSTM
for d = Phase3(1):Phase3(1)+Trainsample
    TrData_B5{d}(:,1:2) = [];
    TrData_B5{d}(:,end) = [];
    TrData_B5{d} = TrData_B5{d}';
end
for d = 1:Phase3(2)-Phase3(1)-Trainsample+1
    Xtest{d}(:,1:2) = [];
    Xtest{d} = Xtest{d}';
end
TrData_B5(Phase1(1):Phase2(2)) = [];
X_training = TrData_B5(1:Trainsample);
Y_training = Capacity_No7(Phase3(1):Phase3(1)+Trainsample-1);
X_validate = Xtest;
Y_validate = Capacity_No7(Phase3(1)+Trainsample:Phase3(2));

%% Train the LSTM model
inputsize = 9-s;
numHiddenUnits1 = 100; numHiddenUnits2 = 100; numResponses = 1;
layers = [ ...
    sequenceInputLayer(inputsize)
    lstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')
    lstmLayer(numHiddenUnits2, 'OutputMode', 'last')
    fullyConnectedLayer(100)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 300; miniBatchSize = 40;
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.005, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',0);
net = trainNetwork(X_training, Y_training', layers, options);

%% Plot the training data
YPred_train =[]; Ytrain = [];
for i = 1 : size(X_training,2)
    yPred_train = predict(net, X_training{i}, 'MiniBatchSize',1);
    YPred_train = [YPred_train yPred_train];
    Ytrain      = [Ytrain Y_training(i)];
end

YPred_validate =[]; Yvalidate = [];
for i = 1 : size(X_validate,2)
    yPred_validate = predict(net, X_validate{i}, 'MiniBatchSize',1);
    YPred_validate = [YPred_validate yPred_validate];
    Yvalidate      = [Yvalidate Y_validate(i)];
end

%% Plot the concerned information for training dataset
figure
subplot(121)
plot(YPred_train,'o-','Linewidth',1.5,'MarkerSize',4);
hold on;
plot(Ytrain,'h--','MarkerFaceColor','r');
xlabel("No. of training cycles");
ylabel("Capacity (Amp-hr)");
legend('Prediction (Proposed)','Real data');
xlim([0, size(Ytrain,2)]);
subplot(122)
bar(YPred_train-Ytrain)
xlabel("No. of training cycles")
ylabel("Error (Amp-hr)")
xlim([0, size(Ytrain,2)])

%% Plot the concerned information for testing dataset
K = 1:size(Yvalidate,2);
figure
subplot(121)
plot(YPred_validate,'o-','Linewidth',1.5,'MarkerSize',4)
hold on
plot(Yvalidate,'h--','MarkerFaceColor','r')
xlabel("No. of testing cycles")
ylabel("Capacity (Amp-hr)");
legend('Prediction (Proposed)','Real data');
xlim([0, size(Yvalidate,2)])
subplot(122)
bar(YPred_validate-Yvalidate)
xlabel("No. of testing cycles")
ylabel("Error (Amp-hr)")
xlim([0, size(Yvalidate,2)])

%% 
m = mean(YPred_validate-Yvalidate);
s = std(YPred_validate-Yvalidate);
RMSE = sqrt(mean((YPred_validate-Yvalidate).^2))/2*100
toc