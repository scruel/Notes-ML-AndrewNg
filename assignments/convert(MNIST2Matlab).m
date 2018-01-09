% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program reads raw MNIST files available at 
% http://yann.lecun.com/exdb/mnist/ 
% and converts them to files in matlab format 
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.  

% This program was originally written by Yee Whye Teh 

% Work with test files first 
fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n'); 

f = fopen('t10k-images-idx3-ubyte','r');
[a,count] = fread(f,4,'int32');
  
g = fopen('t10k-labels-idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n'); 
n = 1000;

Df = cell(1,10);
for d=0:9,
  Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');
end;
  
for i=1:10,
  fprintf('.');
  rawimages = fread(f,28*28*n,'uchar');
  rawlabels = fread(g,n,'uchar');
  rawimages = reshape(rawimages,28*28,n);

  for j=1:n,
    fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
    fprintf(Df{rawlabels(j)+1},'\n');
  end;
end;

fprintf(1,'\n');
for d=0:9,
  fclose(Df{d+1});
  D = load(['test' num2str(d) '.ascii'],'-ascii');
  fprintf('%5d Digits of class %d\n',size(D,1),d);
  save(['test' num2str(d) '.mat'],'D','-mat');
end;


% Work with trainig files second  
f = fopen('train-images-idx3-ubyte','r');
[a,count] = fread(f,4,'int32');

g = fopen('train-labels-idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n'); 
n = 1000;

Df = cell(1,10);
for d=0:9,
  Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
end;

for i=1:60,
  fprintf('.');
  rawimages = fread(f,28*28*n,'uchar');
  rawlabels = fread(g,n,'uchar');
  rawimages = reshape(rawimages,28*28,n);

  for j=1:n,
    fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
    fprintf(Df{rawlabels(j)+1},'\n');
  end;
end;

fprintf(1,'\n');
for d=0:9,
  fclose(Df{d+1});
  D = load(['digit' num2str(d) '.ascii'],'-ascii');
  fprintf('%5d Digits of class %d\n',size(D,1),d);
  save(['digit' num2str(d) '.mat'],'D','-mat');
end;

dos('rm *.ascii');

