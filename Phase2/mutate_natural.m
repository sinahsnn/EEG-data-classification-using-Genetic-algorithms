function y = mutate_natural(y, p)
    [N,M] = size(y);
    inds = rand(N,M) < p;
    values = randi(20, N, M) - 10;
    y = y + inds.* values;
end