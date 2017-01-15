function reduced = grouping(segs)

N = size(segs,1);
marked = zeros(N,1);
groups = {};

for i = 1:N - 1
    if marked(i) == 1
        continue
    end
    
    group = segs(i,:);
    
    nextStart = segs(i,1) + 1;
    j = i + 1;
    while segs(j,1) <= nextStart && j < N
       interval2 = [segs(j,1), segs(j,3)];
       if segs(j,3) == segs(i,3) && segs(j,1) == nextStart
           group = [group; segs(j,:)];
           nextStart = nextStart + 1;
           marked(j) = 1;
       end
       j = j + 1;
    end
    
    groups{end + 1} = group;
end

reduced = zeros(1,4);
for i = 1:size(groups,2)
    chunk = groups{i};
    Nchunk = size(chunk,1);
    if Nchunk == 1
        reduced = [reduced; chunk];
    elseif Nchunk == 2
        reduced = [reduced; chunk(randi(2),:)];
    else
        reduced = [reduced; chunk(1:2:Nchunk,:)];
    end
end

reduced = reduced(1:end,:);