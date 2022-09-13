function idxs = roulette_wheel(fit,N)
N = N/2;
fit = fit/sum(fit);
fit_cdf = cumsum(fit);
idxs = zeros(N,2);
for i = 1:N
    a = rand;
    j = 1;
    while(fit_cdf(j) < a)
        j = j+1;
    end
    idxs(i,1) = j;
    a = rand;
    j = 1;
    while(fit_cdf(j) < a)
        j = j+1;
    end
    idxs(i,2) = j;
end


end