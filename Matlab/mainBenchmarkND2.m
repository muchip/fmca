clear all
close all
maxNumCompThreads(1)
diary('diary2.txt');
logger = 'matlabLogger2';
rparam = 10;
filename = sprintf('%s.txt', logger);
fileID = fopen(filename, 'w');
fprintf(fileID, '      npts       ctim       stim       cond        nzS       dtim       Ltim        nzL\n');
fclose(fileID);
tic
for i = 1:20
    display(i)
    fileID = fopen(filename, 'a');
    tic
    [P,ctime] = MEXmainTestBenchmarkND2(i);
    toc
    fprintf(fileID, '%10d %10.2f', 2^i, ctime);
    tic
    S = tril(sparse(P(:,1),P(:,2),P(:,3)));
    S = S + rparam * speye(size(S));
    clear P;
    S = tril(S) + tril(S,-1)';
    if i <= 10
      cnd = condest(S)
    else
      cnd = 0
    end
    stime = toc
    fprintf(fileID, ' %10.2f %10.2f %10d', stime, cnd, ceil(nnz(S) / size(S,1)));
    tic
    p = dissect(S);
    dtime = toc
    fprintf(fileID, ' %10.2f', dtime);
    tic
    L = chol(S(p,p), 'lower');
    Ltime = toc
    fprintf(fileID, ' %10.2f %10d\n', Ltime, ceil(nnz(L) / size(S,1)));
    whos
end

