clc; clear; close all

%% Load data-Extract the discharging cycles from Battery 5 and 7
load('data/Data_bettery.mat')
DATA_B7 = DATA_No7; Capacity_B7 = Capacity_No7;
DATA_B5 = DATA_No5; Capacity_B5 = Capacity_No5;
DATA_B5(43) = []; Capacity_B5(43) = [];
Phase1 = [1, 30]; Phase2 = [31, 107]; Phase3 = [108, 165];

%% Calculate the minimum length of each phase
[TrainLength3, Trainsample3] = CalLength(DATA_B7, Phase3, 0.7);
[TrainLength1, Trainsample1] = CalLength(DATA_B5, Phase3, 0.7);
TrainLength = min([TrainLength1, TrainLength3]);
Trainsample = min([Trainsample1, Trainsample3]);

%% Obtain the staionary and non-staionary source from training data
tau = 5; m = 3; s = 4; 
[TrData_B7, XtrainB7, YS] = Dataslicing1(DATA_B7, Capacity_B7, Phase3, TrainLength, Trainsample, tau, m);
Xtrain = XtrainB7;
[est_Ps, est_Pn, est_As, est_An, ssa_results] = ssa(Xtrain, s, 'reps', 20, 'equal_epochs', Trainsample, 'random_seed', 12345);
Ts = est_Ps * Xtrain; Tn = est_Pn * Xtrain;
[Tn,meanX,stdX] = autoscale_new(Tn');

%% Arranging the training dataset
Xtrain = []; Xtrain_error = [];
for d = Phase3(1):Phase3(1)+Trainsample-1
       xtrain = [];
       N = size(TrData_B7{d},1);
       xtrain = mapminmax('apply', TrData_B7{d}', YS);
       xtrain_error = est_Ps * xtrain;
       xtrain = est_Pn * xtrain;
       for j = 1:9-s
           xtrain(j,:) = smoothdata(xtrain(j,:),'gaussian',15);
       end
       for j = 1:s
           xtrain_error(j,:) = smoothdata(xtrain_error(j,:),'gaussian',11);
       end
       xtrain = autoscale_new(xtrain',meanX,stdX);
       xtrain = xtrain';
       xtrain(:,231:end) = [];
       xtrain_error(:,231:end) = [];
       xtrain_error(:,1:20) = [];
       N = size(xtrain,2);
       cycleID = 1:N;
       Capacity = Capacity_B7(d)*ones(N,1);
       TrData_B7{d} = [(d-Phase3(1)+1)*ones(N,1) cycleID' xtrain' Capacity];
       Xtrain = [Xtrain; TrData_B7{d}];
       Xtrain_error{d-Phase3(1)+1} = xtrain_error;
end

%% Arranging Testing Dataset
Xtest = []; TT = []; Xtest_D = []; Xtest_Error = [];
for d = Phase3(1):Phase3(2)
    xtest = [];
    N = size(DATA_B5{d},2);
    data = DATA_B5{d}(:,1:TrainLength);
    N = size(data, 2);
    data = [reconstitution(data(1,:), N, m, tau); reconstitution(data(2,:), N, m, tau); reconstitution(data(3,:), N, m, tau)];
    data = mapminmax('apply', data, YS);
    xtest_error = est_Ps * data;
    xtest = est_Pn * data;
    for j = 1:9-s
        xtest(j,:) = smoothdata(xtest(j,:),'gaussian',15);
    end
    for j = 1:s
        xtest_error(j,:) = smoothdata(xtest_error(j,:),'gaussian',11);
    end
    xtest(:,231:end) = [];
    xtest_error(:,231:end) = [];
    xtest_error(:,1:20) = [];
    xtest = autoscale_new(xtest',meanX,stdX);
    xtest = xtest';
    cycleID = 1:size(xtest,2);
    Xtest{d-Phase3(1)+1} = [(d-Phase3(1)+1)*ones(size(xtest,2),1) cycleID' xtest'];
    Xtest_D = [Xtest_D; Xtest{d-Phase3(1)+1}];
    
    TT = [TT xtest];
    Xtest_Error{d-Phase3(1)+1} = xtest_error;
end

%% Arrange the data for the following LSTM
for d = Phase3(1):Phase3(1)+Trainsample-1
    TrData_B7{d}(:, 1:2) = [];
    TrData_B7{d}(:, end) = [];
    TrData_B7{d} = TrData_B7{d}';
end
TrData = TrData_B7(Phase3(1):Phase3(1)+Trainsample-1);
for d = 1: Phase3(2)-Phase3(1)+1
    Xtest{d}(:, 1:2) = [];
    Xtest{d} = Xtest{d}';
end

X_training = TrData; Y_training = Capacity_No7(Phase3(1):Phase3(1)+Trainsample-1);
X_validate = Xtest;  Y_validate = Capacity_No5(Phase3(1):Phase3(2));

%% Train the model
inputsize = 9-s;
numHiddenUnits1 = 100; numHiddenUnits2 = 100; numResponses = 1;
layers = [ ...
    sequenceInputLayer(inputsize)
    lstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')
    lstmLayer(numHiddenUnits2, 'OutputMode', 'last')
    fullyConnectedLayer(50)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 200; miniBatchSize = 4;
options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress',...
    'Verbose', 0);
net = trainNetwork(X_training, Y_training', layers, options);

%% Plot the training data
YPred_train =[]; Ytrain = [];
for i = 1 : size(X_training, 2)
    yPred_train = predict(net, X_training{i}, 'MiniBatchSize',1);
    YPred_train = [YPred_train yPred_train];
    Ytrain      = [Ytrain Y_training(i)];
end

YPred_validate = []; Yvalidate = [];
for i = 1 : size(X_validate, 2)
      yPred_validate = predict(net, X_validate{i}, 'MiniBatchSize',1);
      YPred_validate = [YPred_validate yPred_validate];
      Yvalidate      = [Yvalidate Y_validate(i)];
end

%% Plot the concerned information for training dataset
figure
subplot(121)
plot(YPred_train, 'o-', 'Linewidth', 1.5, 'MarkerSize', 4);
hold on;
plot(Ytrain, 'h--', 'MarkerFaceColor', 'r');
xlabel("No. of training cycles");
ylabel("Capacity (Amp-hr)");
legend('Prediction (Proposed)', 'Real data');
xlim([0, size(Ytrain, 2)]);
subplot(122)
bar(YPred_train-Ytrain)
xlabel("No. of training cycles")
ylabel("Error (Amp-hr)")
xlim([0, size(Ytrain, 2)])

%% Plot the concerned information for testing dataset
K = 1:size(Yvalidate, 2);
figure
subplot(121)
plot(YPred_validate, 'o-','Linewidth', 1.5, 'MarkerSize',4)
hold on
plot(Yvalidate, 'h--', 'MarkerFaceColor', 'r')
xlabel("No. of testing cycles")
ylabel("Capacity (Amp-hr)");
legend('Prediction (Proposed)','Real data');
xlim([0, size(Yvalidate, 2)])
subplot(122)
bar(YPred_validate-Yvalidate)
xlabel("No. of testing cycles")
ylabel("Error (Amp-hr)")
xlim([0, size(Yvalidate, 2)])

%% Calculate the estimation error
m = mean(YPred_validate-Yvalidate);
s = std(YPred_validate-Yvalidate);
RMSE = sqrt(mean((YPred_validate-Yvalidate).^2))/2*100

