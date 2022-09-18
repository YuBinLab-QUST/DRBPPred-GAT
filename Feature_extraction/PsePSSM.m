clear all
clc
lamdashu=2;
WEISHU=3839;
load('xuliechangdu.mat')
for i=1:WEISHU
    nnn=num2str(i);
    name = strcat(nnn,'.pssm');
    fid{i}=importdata(name);
end
c=cell(WEISHU,1);
for t=1:WEISHU
    clear shu d
shu=fid{t}.data;
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
for i=1:WEISHU
[MM,NN]=size(c{i});
for  j=1:20
  x(i,j)=sum(c{i}(:,j))/MM;
  end
end
xx=[];
sheta=[];
shetaxin=[];

for lamda=1:lamdashu;
for t=1:WEISHU
  [MM,NN]=size(c{t});
  clear xx
   for  j=1:20
      for i=1:MM-lamda
       xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
      end
      sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
   end
end
shetaxin=[shetaxin,sheta];
end
psepssm=[x,shetaxin];
csvwrite('psepppssm2.csv',psepssm)
      
