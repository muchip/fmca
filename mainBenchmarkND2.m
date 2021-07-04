clear all
close all
diary('diary2.txt');
logger = 'matlabLogger2';
rparam = 1;
filename = sprintf('%s.txt', logger);
fileID = fopen(filename, 'w');
fprintf(fileID, '      npts       ctim       stim        nzS       dtim       Ltim        nzL\n');
fclose(fileID);
tic
for i = 1:20
    fileID = fopen(filename, 'a');
    tic
    [P,ctime] = MEXmainTestBenchmarkND(i);
    toc
    fprintf(fileID, '%10d %10.2f', 2^i, ctime);
    tic
    S = sparse(P(:,1),P(:,2),P(:,3));
    S = S + rparam * speye(size(S));
    clear P;
    stime = toc
    fprintf(fileID, ' %10.2f %10d', stime, floor(nnz(S) / size(S,1)));
    tic
    p = dissect(S);
    dtime = toc
    fprintf(fileID, ' %10.2f', dtime);
    tic
    L = chol(S(p,p));
    Ltime = toc
    fprintf(fileID, ' %10.2f %10d\n', Ltime, floor(nnz(L) / size(S,1)));
end

