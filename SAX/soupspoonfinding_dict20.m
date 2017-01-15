[a,b]=timeseries2symbol(energy_acc_xyz, 20, 20, 10);
ptn = [ 5     4     4     4     4     4     4     4     4     4     4     4     4     4     4    10    10     7     4     4];
dists = [];

for i = 1: length(a)
     disttmp = min_dist(a(i,:), ptn, 10,4);
     dists = [dists;disttmp];
end

[sdists,sind] = sort(dists);
find(dists == 0)