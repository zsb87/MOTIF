function y=isint(x)
%ISINT(X) Returns logical true when input, X, is an integer.
y=logical(abs(round(x)-x)<eps);