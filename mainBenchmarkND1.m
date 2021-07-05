clear all
close all
diary('diary1.txt');
logger = 'matlabLogger1';
rparam = 1;
filename = sprintf('%s.txt', logger);
fileID = fopen(filename, 'w');
fprintf(fileID, '      npts       ctim       stim       cond        nzS       dtim       Ltim        nzL\n');
fclose(fileID);
tic
for i = 1:20
    display(i)
    fileID = fopen(filename, 'a');
    tic
    [P,ctime] = MEXmainTestBenchmarkND1(i);
    toc
    fprintf(fileID, '%10d %10.2f', 2^i, ctime);
    tic
    S = tril(sparse(P(:,1),P(:,2),P(:,3)));
    S = S + rparam * speye(size(S));
    clear P;
    cnd = condest(S)
    stime = toc
    fprintf(fileID, ' %10.2f %10.2f %10d', stime, cnd, floor(nnz(S) / size(S,1)));
    tic
    p = dissect(S);
    dtime = toc
    fprintf(fileID, ' %10.2f', dtime);
    tic
    L = chol(S(p,p), 'lower');
    Ltime = toc
    fprintf(fileID, ' %10.2f %10d\n', Ltime, floor(nnz(L) / size(S,1)));
    whos
end

