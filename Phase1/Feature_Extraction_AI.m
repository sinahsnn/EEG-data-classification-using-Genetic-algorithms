%% Loading the Dataset
load('CI_Project_data.mat');
%% Extracting Features
Train_data_len = 165;
Fs = 256; %Hz
channel_len = 256;
Features_len = 3000;
w = linspace(0, Fs/2 - 1, channel_len /2)'; % 0 - pi
Train_Features = zeros(Features_len ,Train_data_len) ;
for i = 1:Train_data_len
    A = TrainData(:,:,i);
    
    
    % variance of all channels
    Train_Features(1:30, i) = var(A, [], 2); 
    
    % FF ( Form Factor ) for all channels
    diff_A = diff(A, 1, 2); 
    diff2_A = diff(A, 2, 2);
    Train_Features(31:60, i) = std(A, [], 2).* std(diff2_A, [], 2)./ std(diff_A, [], 2).^2; 
    
    % argmax of the maximum frequency
    A_fft = abs(fft(A, channel_len, 2));
    [max_eng, max_freq_ind] = max(A_fft(:,1:channel_len/2), [], 2);
    Train_Features(61:90, i) = max_freq_ind ;  % mapping 0-\pi to 0-128 Hz
    
    % mean frequency
    mean_Freq = A_fft(:,1:channel_len/2)*w/sum(w);
    Train_Features(91:120, i) = mean_Freq;  
    
    % median frequency
    Train_Features(121:150, i) = medfreq(A',Fs);
    
    % relative energy of frequency bands
    total_energy = sum(A_fft(:, 3:51), 2);
    delta = sum(A_fft(:, 3:9), 2)./total_energy;
    alphai = sum(A_fft(:, 10:16), 2)./total_energy;
    beta1 = sum(A_fft(:, 17:23), 2)./total_energy;
    beta2 = sum(A_fft(:, 24:30), 2)./total_energy;
    gamma1 = sum(A_fft(:, 31:37), 2)./total_energy;
    gamma2 = sum(A_fft(:, 38:44), 2)./total_energy;
    gamma3 = sum(A_fft(:, 45:51), 2)./total_energy;
    
    Train_Features(151:180, i) = delta;
    Train_Features(181:210, i) = alphai;
    Train_Features(211:240, i) = beta1;
    Train_Features(241:270, i) = beta2;
    Train_Features(271:300, i) = gamma1;
    Train_Features(301:330, i) = gamma2;
    Train_Features(331:360, i) = gamma3;
    
    
    % histogram
    xbin = linspace(-15, 15, 11);
    
    for j = 1:30
        Train_Features(361 + (j-1)*11 : 371 + (j-1)*11, i) = hist(A(j, :), xbin)';
    end
    
    Train_Features(791:820, i) = max_eng;
    %  pwelch psd energy
    for j = 1:30
       t = abs(pwelch(A(j,:)));
       [~,ind_max] = max(t);
       
        total_energy = sum(t(43:86, :), 1);
        delta = sum(t([61:63, 66:68], :), 1)./total_energy;
        alpha = sum(t([58:60, 69:71], :), 1)./total_energy;
        beta1 = sum(t([55:57, 72:74], :), 1)./total_energy;
        beta2 = sum(t([52:54, 75:77], :), 1)./total_energy;
        gamma1 = sum(t([49:51, 78:80], :), 1)./total_energy;
        gamma2 = sum(t([46:48, 81:83], :), 1)./total_energy;
        gamma3 = sum(t([43:45, 84:86], :), 1)./total_energy;
       
        Train_Features(821 + (j-1)*45/5 : 820 + j*45/5, i) = [ind_max, total_energy,...
            delta, alpha, beta1, beta2, gamma1, gamma2, gamma3]';    
    end
    % asymetric index 
    h = 1090;
    alphai = log(alphai);
    for j = 1 : 30
        for k = j+1 : 30
            h = h + 1;
            Train_Features(h, i) = alphai(j)-alphai(k);
            
        end
    end
    
    % bandpower features
    y6 = 1526;
    for j = 1:30
       Train_Features(y6, i) = bandpower(A(j,:), Fs, [2 8]);
       Train_Features(y6+1, i) = bandpower(A(j,:), Fs, [9 15]);
       Train_Features(y6+2, i) = bandpower(A(j,:), Fs, [16 22]);
       Train_Features(y6+3, i) = bandpower(A(j,:), Fs, [23 29]);
       Train_Features(y6+4, i) = bandpower(A(j,:), Fs, [30 36]);
       Train_Features(y6+5, i) = bandpower(A(j,:), Fs, [37 43]);
       Train_Features(y6+6, i) = bandpower(A(j,:), Fs, [44 50]);
       
       y6 = y6 + 7;
    end
    y7 = 1741;
    for j = 1:30
        tmp1 = arburg(A(j,:), 5);
        Train_Features(y7:y7+3,i) = tmp1(2:5)';
        y7 = y7 + 4;
    end
    y8 = 1862;
    for j = 1:30
        Train_Features(y8:y8+29) = kurtosis(A, [], 2);
    end

    %correlation
    y9 = 1900;
    for j = 1 : 30
        for k = j+1 : 30
            y9 = y9 + 1;
            Train_Features(y9, i) = corr(A_fft(j,:)',A_fft(k,:)');
            
        end
    end
    
end

%% Normalization 
[Normalized_Train_Features,xPS] = mapminmax(Train_Features) ;

% Label Indices
Img_Mov_indices = find(TrainLabel==1) ;
Mental_Arith_indices = find(TrainLabel==0) ;

%% Fischer Feature Selection

J = zeros(Features_len,1);
for i = 1:Features_len
    u1 = mean(Normalized_Train_Features(i,Img_Mov_indices)) ;
    S1 = (Normalized_Train_Features(i,Img_Mov_indices)-u1)*(Normalized_Train_Features(i,Img_Mov_indices)-u1)' ; % =var(Normalized_Train_Features(i,PVC_indices))
    u2 = mean(Normalized_Train_Features(i,Mental_Arith_indices)) ;
    S2 = (Normalized_Train_Features(i,Mental_Arith_indices)-u2)*(Normalized_Train_Features(i,Mental_Arith_indices)-u2)' ; % =var(Normalized_Train_Features(i,Normal_indices))
    Sw = S1/length(Img_Mov_indices)+S2/length(Mental_Arith_indices) ;
    
    u0 = mean(Normalized_Train_Features(i,:)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    
    J(i) = Sb/Sw ;
end

% saver normailzation parameters and features
save('Train_Features','Normalized_Train_Features','xPS')



%% Choose the 10 best Features for MLP and RBF
[mxx, ind] = maxk(J, 10);

figure
plot3(Normalized_Train_Features(ind(1),Img_Mov_indices),Normalized_Train_Features(ind(2),Img_Mov_indices),Normalized_Train_Features(ind(3),Img_Mov_indices),'*r') ;
hold on
plot3(Normalized_Train_Features(ind(1),Mental_Arith_indices),Normalized_Train_Features(ind(2),Mental_Arith_indices),Normalized_Train_Features(ind(3),Mental_Arith_indices),'og') ;
title('Fetures #1, #2, #3') ;
grid on

%% MLP - Hidden Neuron Number Comparison
ACCMat_MLP = zeros(1,25);
for N = 1:1:25

    total_err = 0 ; 
    % using 5-fold cross-validation
    for k=1:5
        train_indices = [1:(k-1)*33,k*33+1:165] ;
        valid_indices = (k-1)*33+1:k*33 ;

        TrainX = Normalized_Train_Features(ind,train_indices) ;
        ValX = Normalized_Train_Features(ind,valid_indices) ;
        TrainY = TrainLabel(:,train_indices) ;
        ValY = TrainLabel(:,valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet(N);
        net = train(net,TrainX,TrainY);
%        net.layers{1}.transferFcn = 'logsig';
%        net.layers{3}.transferFcn = 'purelin';
        
        predict_y = net(ValX);
        predictedLabel = predict_y > 0.5;
        err = sum(abs(predictedLabel - ValY));
        total_err = total_err + err;
        
    end

    ACCMat_MLP(N) = 1 - total_err / 165;
end
%% MLP - Comparing Differnet Activation Functions
ACCMat_activ = zeros(1,5);
activations = ['logsig';'tansig';'logsig'; 'pureli';'poslin'];
for N = 1:5

    total_err = 0 ; 
    % using 5-fold cross-validation
    for k=1:5
        train_indices = [1:(k-1)*33,k*33+1:165] ;
        valid_indices = (k-1)*33+1:k*33 ;

        TrainX = Normalized_Train_Features(ind,train_indices) ;
        ValX = Normalized_Train_Features(ind,valid_indices) ;
        TrainY = TrainLabel(:,train_indices) ;
        ValY = TrainLabel(:,valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet(20);
        net = train(net,TrainX,TrainY);
        net.layers{1}.transferFcn = 'tansig';
%        net.layers{3}.transferFcn = 'purelin';
        
        predict_y = net(ValX);
        predictedLabel = predict_y > 0.5;
        err = sum(abs(predictedLabel - ValY));
        total_err = total_err + err;
        
    end

    ACCMat_activ(N) = 1 - total_err / 165;
end

%% MLP - 2 Hidden Neuron Number Comparison
ACCMat_MLP_2layer = zeros(1,25);
for N = 1:1:25

    total_err = 0 ; 
    % using 5-fold cross-validation
    for k=1:5
        train_indices = [1:(k-1)*33,k*33+1:165] ;
        valid_indices = (k-1)*33+1:k*33 ;

        TrainX = Normalized_Train_Features(ind,train_indices) ;
        ValX = Normalized_Train_Features(ind,valid_indices) ;
        TrainY = TrainLabel(:,train_indices) ;
        ValY = TrainLabel(:,valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet([N ceil(N/3)]);
        net = train(net,TrainX,TrainY);
%        net.layers{1}.transferFcn = 'logsig';
%        net.layers{3}.transferFcn = 'purelin';
        
        predict_y = net(ValX);
        predictedLabel = predict_y > 0.5;
        err = sum(abs(predictedLabel - ValY));
        total_err = total_err + err;
        
    end

    ACCMat_MLP_2layer(N) = 1 - total_err / 165;
end



%% RBF - Comparison of Differnent Neuron Numbers and Spread Values
ACCMat_RBF = zeros(6,6);
spreadMat = [.1,.5,.9,1.5,2, 2.5] ;
NMat = [5,10,15,20,25,30,50] ;
for s = 1:6
    spread = spreadMat(s) ;
    for n = 1:7 
        Maxnumber = NMat(n) ;
        ACC = 0 ;
        % 5-fold cross-validation
        for k=1:5
            train_indices = [1:(k-1)*33,k*33+1:165] ;
            valid_indices = (k-1)*33+1:k*33 ;

            TrainX = Normalized_Train_Features(ind,train_indices) ;
            ValX = Normalized_Train_Features(ind,valid_indices) ;
            TrainY = TrainLabel(:,train_indices) ;
            ValY = TrainLabel(:,valid_indices) ;

            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber, 10) ;
            predict_y = net(ValX);
            
            Thr = 0.5 ;
            predict_y = predict_y >= Thr ;

            ACC = ACC + 35 - sum(abs(predict_y - ValY)) ;
        end
        ACCMat_RBF(s,n) = ACC/165 ;
    end
end




%% Extracting Test Features

Test_data_len = 45;
Fs = 256; %Hz
channel_len = 256;
Features_len = 3000;
w = linspace(0, Fs/2 - 1, channel_len /2)'; % 0 - pi
Test_Features = zeros(Features_len ,Test_data_len) ;
for i = 1:Test_data_len
    A = TestData(:,:,i);
    
    
    % variance of all channels
    Test_Features(1:30, i) = var(A, [], 2); 
    
    % FF ( Form Factor ) for all channels
    diff_A = diff(A, 1, 2); 
    diff2_A = diff(A, 2, 2);
    Test_Features(31:60, i) = std(A, [], 2).* std(diff2_A, [], 2)./ std(diff_A, [], 2).^2; 
    
    % argmax of the maximum frequency
    A_fft = abs(fft(A, channel_len, 2));
    [max_eng, max_freq_ind] = max(A_fft(:,1:channel_len/2), [], 2);
    Test_Features(61:90, i) = max_freq_ind ;  % mapping 0-\pi to 0-128 Hz
    
    % mean frequency
    mean_Freq = A_fft(:,1:channel_len/2)*w/sum(w);
    Test_Features(91:120, i) = mean_Freq;  
    
    % median frequency
    Test_Features(121:150, i) = medfreq(A',Fs);
    
    % relative energy of frequency bands
    total_energy = sum(A_fft(:, 3:51), 2);
    delta = sum(A_fft(:, 3:9), 2)./total_energy;
    alphai = sum(A_fft(:, 10:16), 2)./total_energy;
    beta1 = sum(A_fft(:, 17:23), 2)./total_energy;
    beta2 = sum(A_fft(:, 24:30), 2)./total_energy;
    gamma1 = sum(A_fft(:, 31:37), 2)./total_energy;
    gamma2 = sum(A_fft(:, 38:44), 2)./total_energy;
    gamma3 = sum(A_fft(:, 45:51), 2)./total_energy;
    
    Test_Features(151:180, i) = delta;
    Test_Features(181:210, i) = alphai;
    Test_Features(211:240, i) = beta1;
    Test_Features(241:270, i) = beta2;
    Test_Features(271:300, i) = gamma1;
    Test_Features(301:330, i) = gamma2;
    Test_Features(331:360, i) = gamma3;
    
    
    % histogram
    xbin = linspace(-15, 15, 11);
    
    for j = 1:30
        Test_Features(361 + (j-1)*11 : 371 + (j-1)*11, i) = hist(A(j, :), xbin)';
    end
    
    Test_Features(791:820, i) = max_eng;
    %  pwelch psd energy
    for j = 1:30
       t = abs(pwelch(A(j,:)));
       [~,ind_max] = max(t);
       
        total_energy = sum(t(43:86, :), 1);
        delta = sum(t([61:63, 66:68], :), 1)./total_energy;
        alpha = sum(t([58:60, 69:71], :), 1)./total_energy;
        beta1 = sum(t([55:57, 72:74], :), 1)./total_energy;
        beta2 = sum(t([52:54, 75:77], :), 1)./total_energy;
        gamma1 = sum(t([49:51, 78:80], :), 1)./total_energy;
        gamma2 = sum(t([46:48, 81:83], :), 1)./total_energy;
        gamma3 = sum(t([43:45, 84:86], :), 1)./total_energy;
       
        Test_Features(821 + (j-1)*45/5 : 820 + j*45/5, i) = [ind_max, total_energy,...
            delta, alpha, beta1, beta2, gamma1, gamma2, gamma3]';    
    end
    % asymetric index 
    h = 1090;
    alphai = log(alphai);
    for j = 1 : 30
        for k = j+1 : 30
            h = h + 1;
            Test_Features(h, i) = alphai(j)-alphai(k);
            
        end
    end
    
    % bandpower features
    y6 = 1526;
    for j = 1:30
       Test_Features(y6, i) = bandpower(A(j,:), Fs, [2 8]);
       Test_Features(y6+1, i) = bandpower(A(j,:), Fs, [9 15]);
       Test_Features(y6+2, i) = bandpower(A(j,:), Fs, [16 22]);
       Test_Features(y6+3, i) = bandpower(A(j,:), Fs, [23 29]);
       Test_Features(y6+4, i) = bandpower(A(j,:), Fs, [30 36]);
       Test_Features(y6+5, i) = bandpower(A(j,:), Fs, [37 43]);
       Test_Features(y6+6, i) = bandpower(A(j,:), Fs, [44 50]);
       
       y6 = y6 + 7;
    end
    y7 = 1741;
    for j = 1:30
        tmp1 = arburg(A(j,:), 5);
        Test_Features(y7:y7+3,i) = tmp1(2:5)';
        y7 = y7 + 4;
    end
    y8 = 1862;
    for j = 1:30
        Test_Features(y8:y8+29) = kurtosis(A, [], 2);
    end

    %correlation
    y9 = 1900;
    for j = 1 : 30
        for k = j+1 : 30
            y9 = y9 + 1;
            Test_Features(y9, i) = corr(A_fft(j,:)',A_fft(k,:)');
            
        end
    end
    
end

%% Tests on MLP

% Normalization
Normalized_Test_Features = mapminmax('apply',Test_Features,xPS) ;

% Classification
N = 22 ; % Best parameter found in training step
TrainX = Normalized_Train_Features(ind, :) ;
TrainY = TrainLabel ;
TestX = Normalized_Test_Features(ind, :) ;
%TestY = New_Test_Label ; 

net = patternnet(N);
net = train(net,TrainX,TrainY);

predict_y = net(TestX);
Test_MLP_Result = predict_y > 0.5;

% saver normailzation parameters and features
%save('MLP_Results','Test_MLP_Result')

     

%%
%% Tests on RBF

spread = 2;
Maxnumber = 10;
TrainX = Normalized_Train_Features(ind, :) ;
TrainY = TrainLabel ;
TestX = Normalized_Test_Features(ind, :) ;
%TestY = New_Test_Label ; 

net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber, 10) ;
predict_y = net(TestX);
Thr = 0.5 ;
Test_RBF_Result = predict_y > Thr ;
%save('RBF_Results','Test_RBF_Result')


