clear all
clc
% [data,A,B]=xlsread('dataset.xlsx','pos');
% AA=A(2:end,4);
input=importdata('PDB186.txt');
data=input(1:end,:);
[m,n]=size(data);
vector=[];
for i=1:m;
 vector= [vector;EBGW_demo(data{i})];
end

% save EBGW_pos_31.mat vector
ebgw=vector;
csvwrite('EBGW_PDB186.csv',ebgw)

