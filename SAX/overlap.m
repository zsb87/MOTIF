function intersect = overlap(a, b)

    assert(a(1) <= b(1),'first start must not be larger than second start');

    minElement = min([a(1), a(2), b(1), b(2)]);
    maxElement = max([a(1), a(2), b(1), b(2)]);

    if (a(2) < b(1))
        intersect = 0;
    elseif (a(2) >= b(2))
        intersect = b(2) - b(1) + 1;
    else
        intersect = a(2) - b(1) + 1;
    end
    intersect = intersect/(maxElement - minElement + 1);

end