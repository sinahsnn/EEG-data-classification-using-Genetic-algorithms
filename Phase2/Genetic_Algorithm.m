
%%
load('Train_Features.mat');
load('Test_Features.mat');
%% Fischer Feature Selection
Img_Mov_indices = find(TrainLabel==1) ;
Mental_Arith_indices = find(TrainLabel==0) ;
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

%% Genetic Algorithm
lenFeatures = 15;
Num_of_features = 100;
[mxx, ind] = maxk(J, Num_of_features);
ChosenFeatures = Normalized_Train_Features(ind, :);

%%
N = 20; % Generation Size
p = 0.03; % mutation prob
max_gen = 1e2;

%initialize the population
P = zeros(N, lenFeatures);
P2 = zeros(N, lenFeatures);


for i = 1:N
    P(i,:) = randperm(Num_of_features, lenFeatures);
    P2(i,:) = randperm(Num_of_features, lenFeatures);
end
P = sort(P,2);
P2 = sort(P2,2);
P_best = P(N,:);
gen = 1;
fit_best_gen = 0;
fitbest = zeros(max_gen,1);
%%
fit = zeros(N, 1);
while gen < max_gen
   for i = 1:N 
       fit(i) = mlp_fitness(ChosenFeatures(P(i,:), :),  TrainLabel);
   end
   [tmp, idx] = max(fit);
   if(fit_best_gen < tmp)
       P_best = P(idx,:);
       fit_best_gen = tmp;
       fitbest(gen,1) = fit_best_gen;
   end
   % roulette-wheel selection
   couples = roulette_wheel(fit - min(fit) - 0.01/(1+gen),N-4);
   % Mutation
   P = mutate_natural(P, p);
   % Cross-over
   P2(1:N-4, :) = crossover_1Point_correction(P, couples);
   
   % Correction
   
   P2(N,:) = P_best;
   P2(N-3:N-1,:) = P(roulette_wheel2(fit,3),:);
   P = P2;
   
   P = correction(P, Num_of_features); 

   
   disp(fit_best_gen);disp(P_best);disp(fit);
   gen = gen + 1; 
   if(gen < 10)
       p = 0.02;
   elseif(gen < 20)
       p = 0.01;
   elseif(gen < 30)
       p = 0.005;
   end
end

%%
%% MLP - Training GA

ind2 = P_best;

total_err = 0 ; 
% using 5-fold cross-validation
for k=1:5
    train_indices = [1:(k-1)*33,k*33+1:165] ;
    valid_indices = (k-1)*33+1:k*33 ;

    TrainX = ChosenFeatures(ind2,train_indices) ;
    ValX = ChosenFeatures(ind2,valid_indices) ;
    TrainY = TrainLabel(:,train_indices) ;
    ValY = TrainLabel(:,valid_indices) ;

    % feedforwardnet, newff, paternnet
    % patternnet(hiddenSizes,trainFcn,performFcn)
    net = patternnet(20);
    net = train(net,TrainX,TrainY);
%       net.layers{1}.transferFcn = 'tansig';
%        net.layers{3}.transferFcn = 'purelin';

    predict_y = net(ValX);
    predictedLabel = predict_y > 0.5;
    err = sum(abs(predictedLabel - ValY));
    total_err = total_err + err;

end

accuracy_mlp_ga = 1 - total_err / 165;
disp(accuracy_mlp_ga);


%% RBF - Training GA
ind2 = P_best;
spread = 2;
Maxnumber = 6;
err = 0 ; 

% using 5-fold cross-validation
for k = 1:5
    train_indices = [1:(k-1)*33,k*33+1:165] ;
    valid_indices = (k-1)*33+1:k*33 ;

    TrainX = ChosenFeatures(ind2,train_indices) ;
    ValX = ChosenFeatures(ind2,valid_indices) ;
    TrainY = TrainLabel(:,train_indices) ;
    ValY = TrainLabel(:,valid_indices) ;



    net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber, 2) ;
    predict_y = net(ValX);
    Thr = 0.5 ;
    predict_y = predict_y > Thr ;
    err = err + sum(abs(predict_y - ValY)) ;
end

accuracy_rbf_ga = 1 - err / 165;



%% Tests on MLP

% Normalization
%Normalized_Test_Features = mapminmax('apply',Test_Features,xPS) ;
ChosenFeatures_Test = Normalized_Test_Features(ind, :);

% Classification
N = 22 ; % Best parameter found in training step
TrainX = ChosenFeatures(ind2, :) ;
TrainY = TrainLabel ;
TestX = ChosenFeatures_Test(ind2, :) ;
%TestY = New_Test_Label ; 

net = patternnet(N);
net = train(net,TrainX,TrainY);

predict_y = net(TestX);
Test_MLP_GA_Result = predict_y > 0.5;

% saver normailzation parameters and features
%save('MLP_GA_Results','Test_MLP_GA_Result')

     

%%
%% Tests on RBF

spread = 2;
Maxnumber = 8;
TrainX = Normalized_Train_Features(ind, :) ;
TrainY = TrainLabel ;
TestX = Normalized_Test_Features(ind, :) ;
%TestY = New_Test_Label ; 

net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber, 2) ;
predict_y = net(TestX);
Thr = 0.5 ;
Test_RBF_GA_Result = predict_y > Thr ;
%save('RBF_GA_Results','Test_RBF_GA_Result')


