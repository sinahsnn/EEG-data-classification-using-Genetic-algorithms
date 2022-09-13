function P = correction(P, Num_of_features)
[N,M]= size(P);
P = sort(P, 2);    
for i = 1:N
    if(P(i,1) <= 0 || P(i,1) == P(i,2))
        P(i,1) = ceil(P(i,2) / 2);
    end
    for j = 2:M-1
        if P(i,j) ==  P(i,j+1)
            P(i,j) = floor((P(i,j-1) + P(i,j+1))/2);
        end
    end
end
for i = 1:N
    if P(i,M) > Num_of_features
        P(i,M) = 2*Num_of_features - P(i,M);
    end
end

end