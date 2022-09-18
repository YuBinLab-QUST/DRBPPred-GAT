clear all
clc
lamdashu=2;
%读取蛋白质序列的pssm
WEISHU=3839;
load('xuliechangdu.mat')
for i=1:WEISHU
    nnn=num2str(i);
    name = strcat(nnn,'.pssm');
    fid{i}=importdata(name);
end
%所有蛋白质序列归一化
c=cell(WEISHU,1);
for t=1:WEISHU
    clear shu d
shu=fid{t}.data;
%知道每条蛋白质的数量，提取出来的矩阵，注意蛋白质的顺序
% shuju=shu(1:i,1:20);
[M,N]=size(shu);
shuju=shu(1:lensec(1,t),1:20);
d=[];
%归一化
for i=1:lensec(1,t)
   for j=1:20
       d(i,j)=1/(1+exp(-shuju(i,j)));
   end
end
c{t}=d(:,:);
end
% %生成PSSM-AAC,x是一个,
% for i=1:WEISHU
% [MM,NN]=size(c{i});
%  for  j=1:20
%    x(i,j)=sum(c{i}(:,j))/MM;
%  end
% end
% %PsePSSM后20*lamda
% xx=[];
% sheta=[];
% shetaxin=[];
% % lamda=1;
% for lamda=1:lamdashu;
% for t=1:WEISHU
%   [MM,NN]=size(c{t});
%   clear xx
%    for  j=1:20
%       for i=1:MM-lamda
%        xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
%       end
%       sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
%    end
% end
% shetaxin=[shetaxin,sheta];
% end
% psepssm=[x,shetaxin];
% csvwrite('psepppssm2.csv',psepssm)
% % save psepppssm10.mat psepssm
      