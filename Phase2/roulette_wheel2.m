function idxs = roulette_wheel2(fit,N)
fit = fit/sum(fit);
fit_cdf = cumsum(fit);
idxs = zeros(N,1);
for i = 1:N
    a = rand;
    j = 1;
    while(fit_cdf(j) < a)
        j = j+1;
    end
    idxs(i,1) = j;
end


end