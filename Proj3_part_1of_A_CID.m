T = 10;
tstep = 1e-2;
t = 0:tstep:T;
delta = 1e-2;
X = 2;
xstep = 1e-3;
x = 0:xstep:X;

tnum = length(t);
xnum = length(x);
u = zeros(tnum, xnum);

%%initial
delta_in = (ab