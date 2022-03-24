function Data=reconstitution(data,N,m,tau)
%该函数用来重构相空间
% m为嵌入空间维数
% tau为时间延迟
% data为输入时间序列
% N为时间序列长度
% Data为输出,是m*n维矩阵
M=N-(m-1)*tau; %相空间中点的个数
Data=zeros(m,M);
for j=1:M
  for i=1:m           %相空间重构
    %Data(i,:)=data(((i-1)*tau+1):1:((i-1)*tau+M));
    Data(i,j)=data((i-1)*tau+j);
  end
end
end