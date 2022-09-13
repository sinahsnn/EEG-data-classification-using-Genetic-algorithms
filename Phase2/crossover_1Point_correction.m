function P = crossover_1Point_correction(Generation, couples)
M = size(Generation, 2);
N = size(couples,1);
P = zeros(2*N, M);
for i = 1:N
    a = couples(i,1);
    b = couples(i,2);
    pivot = randi(M-1);
    
    P(i*2-1,:) = [Generation(a,1:pivot), Generation(b,pivot+1:M)];    
    P(i*2,:) = [Generation(b,1:pivot), Generation(a,pivot+1:M)];
end

end