function Data=reconstitution(data,N,m,tau)
%�ú��������ع���ռ�
% mΪǶ��ռ�ά��
% tauΪʱ���ӳ�
% dataΪ����ʱ������
% NΪʱ�����г���
% DataΪ���,��m*nά����
M=N-(m-1)*tau; %��ռ��е�ĸ���
Data=zeros(m,M);
for j=1:M
  for i=1:m           %��ռ��ع�
    %Data(i,:)=data(((i-1)*tau+1):1:((i-1)*tau+M));
    Data(i,j)=data((i-1)*tau+j);
  end
end
end