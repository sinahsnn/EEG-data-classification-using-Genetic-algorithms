function acc = mlp_fitness(ChosenFeatures, TrainLabel)
%load('Tf.mat')
 total_err = 0 ; 
    % 5-fold cross-validation
    %for k = 1:5
    t = randperm(165);
        train_indices = t(36:end);%[1:(k-1)*33,k*33+1:165] ;
        valid_indices = t(1:35);%(k-1)*33+1:k*33 ;

        TrainX = ChosenFeatures(:,train_indices) ;
        ValX = ChosenFeatures(:,valid_indices) ;
        TrainY = TrainLabel(:,train_indices) ;
        ValY = TrainLabel(:,valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet(19);
        net = train(net,TrainX,TrainY);
%        net.layers{1}.transferFcn = 'logsig';
%        net.layers{3}.transferFcn = 'purelin';
            
        predict_y = net(ValX);
        predictedLabel = predict_y > 0.5;
        err = sum(abs(predictedLabel - ValY));
        total_err = total_err + err;
        
    %end

    acc = 1 - total_err / 35;
    
end