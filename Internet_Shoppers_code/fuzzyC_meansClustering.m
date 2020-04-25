clc;
close all;
clear all;
%data=load('iris1.txt');
%data=data(:,1:end-1);
%d=data(1:2,:)
%d1=data(51:52,:)
%d2=data(101:102,:)
%data=[d;d1;d2];
data1=load('Heir.txt');
dat=csvread('HT.csv');
data=dat*data1;
[row,column]=size(data);
c=3;
%p1=round(rand(150,3));
p1=zeros(30,1)
p2=ones(30,1)
pl1=[p1;p2;p1;p2;p1;p1;p2;p1;p2;p1];
pl1=[pl1;0;0;0;1]
pl2=[p1;p1;p2;p1;p1;p1;p1;p2;p1;p1;0;1;1;0];
pl3=[p2;p1;p1;p1;p2;p2;p1;p1;p1;p2;1;0;0;0];
p=[pl1';pl2';pl3']
centre=cent(p,data,c);
centre1=centre+1;
i=0;
while(centre~=centre1)
    if(i<25)
        dist=fuzzydist(centre,data);
        p1=update1(p,dist);
        [row1,column1]=size(p1);
        centre1=centre;
        centre=cent(p1',data,c);
        i=i+1;
    else
        break;
    end
end
for i=1:row1
    m=p1(i,1);
    for j=1:column1
        if(p1(i,j)>m)
            m=p1(i,j);
        end
    end
    for l=1:column1
        if(p1(i,l)==m)
            p1(i,l)=1;
        else
            p1(i,l)=0;
        end
    end
end
disp('final classification')
disp(p1)
d=p1;
















for i=1:length(d)
    if(d(i,1)==1)
        data(i,5)=0;
    elseif(d(i,2)==1);
        data(i,5)=1;
    else
        data(i,5)=2;
    end
end

train=data(1:0.7*end,:);
test=data(.7*end:end,:);














plp=d;
train=data(1:0.7*end,:);
%data=load('iris1.txt');
[row121,column121]=size(train);
x=train(:,1:column121-1);
y=train(:,column121);
%r = randperm(150, .3*row)'
%test=d(r,:);
v=1;
e11(:,1)=find(y==0);
e12(:,1)=find(y==1);
e23(:,1)=find(y==2);
w1=x(e11(:,1),:);
w2=x(e12(:,1),:);
w3=x(e23(:,1),:);
p=[length(find(y==0)),length(find(y==1)),length(find(y==2))];
p=p./row121;
mean1=sum(w1)./length(w1);
mean2=sum(w2)./length(w2);
mean3=sum(w3)./length(w3);
var=w1;
for i=1:length(w1)
    var(i,:)=w1(i,:)-mean1;
end
std1=sqrt((sum(var.^2))/length(w2));

var=w2;
for i=1:length(w2)
    var(i,:)=w2(i,:)-mean2;
end
std2=sqrt((sum(var.^2))/length(w1));

var=w3;
for i=1:length(w3)
    var(i,:)=w3(i,:)-mean3;
end
std3=sqrt((sum(var.^2))/length(w2));

y111=[];
for i=1:row121
    x1=data(i,1:column121-1);
    p0=p(1)*postprob(x1(1),mean1(1),std1(1))*postprob(x1(2),mean1(2),std1(2))*postprob(x1(3),mean1(3),std1(3))*postprob(x1(4),mean1(4),std1(4));
    p1=p(2)*postprob(x1(1),mean2(1),std2(1))*postprob(x1(2),mean2(2),std2(2))*postprob(x1(3),mean2(3),std2(3))*postprob(x1(4),mean2(4),std2(4));
    p2=p(3)*postprob(x1(1),mean3(1),std3(1))*postprob(x1(2),mean3(2),std3(2))*postprob(x1(3),mean3(3),std3(3))*postprob(x1(4),mean3(4),std3(4));
    if(p0>p1&&p0>p2)
        y111(i)=0;
    elseif(p1>p0&&p1>p2)
        y111(i)=1;
    else
        y111(i)=2;
    end
end















d=size(train)
d=d(2)
 m=length(train);
 x=train(:,1:(d-1));
 y=train(:,d);
 for k=1:length(y)
     if(y(k)==2)
         y(k)=1;
     end
 end
 x1=ones(m,1)
 x=[x1,x];
theta=zeros(d,1);
alpha=.01;
for i=1:50000
    theta(1)=theta(1)-(alpha/m)*(sum((1./(1+exp(-x*theta)))-y));
    theta(2)=theta(2)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,2)));
    theta(3)=theta(3)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,3)));
    theta(4)=theta(4)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,4)));
    theta(5)=theta(5)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,5)));
end
w=theta
 nTest = size(x,1);
    res = zeros(nTest,1);
    for i = 1:nTest
        sigm(i) = sigmoid([x(i,:)] * w);
        %if sigm >= 0.5
         %   res(i) = 1;
        %else
         %   res(i) = 0;
        %end
    end
%errors = abs(y - res);
%err = sum(errors)
%percentage = (1 - err / size(x, 1))*100

 x=train(:,1:(d-1));
 y=train(:,d);
 for k=1:length(y)
     if(y(k)==2)
         y(k)=0;
     end
 end
 x1=ones(m,1)
 x=[x1,x];
theta=zeros(d,1);
alpha=.01;
for i=1:50000
    theta(1)=theta(1)-(alpha/m)*(sum((1./(1+exp(-x*theta)))-y));
    theta(2)=theta(2)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,2)));
    theta(3)=theta(3)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,3)));
    theta(4)=theta(4)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,4)));
    theta(5)=theta(5)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,5)));
end
w=theta
 nTest = size(x,1);
    res = zeros(nTest,1);
    for i = 1:nTest
        sigm1(i) = sigmoid([x(i,:)] * w);
        %if sigm >= 0.5
         %   res(i) = 1;
        %else
         %   res(i) = 0;
        %end
    end

    
    
    
    
 x=train(:,1:(d-1));
 y=train(:,d);
 for k=1:length(y)
     if(y(k)==2)
         y(k)=1;
     else
        y(k)=0; 
     end
 end
 x1=ones(m,1)
 x=[x1,x];
theta=zeros(d,1);
alpha=.01;
for i=1:50000
    theta(1)=theta(1)-(alpha/m)*(sum((1./(1+exp(-x*theta)))-y));
    theta(2)=theta(2)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,2)));
    theta(3)=theta(3)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,3)));
    theta(4)=theta(4)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,4)));
    theta(5)=theta(5)-(alpha/m)*(sum(((1./(1+exp(-x*theta)))-y).*x(:,5)));
end
w=theta
 nTest = size(x,1);
    res = zeros(nTest,1);
    for i = 1:nTest
        sigm2(i) = sigmoid([x(i,:)] * w);
        %if sigm >= 0.5
         %   res(i) = 1;
        %else
         %   res(i) = 0;
        %end
    end
    sig=[sigm',sigm1',sigm2']
    y1=[];
for i=1:length(sig)
    if(sig(i,1)>sig(i,2)&&sig(i,1)>sig(i,3))
        y1(i)=0;
    elseif(sig(i,2)>sig(i,1)&&sig(i,2)>sig(i,3))
        y1(i)=1;
    else
        y1(i)=2;
    end
end
errors = abs(train(:,5) - y1');
err = sum(errors);
logistic_percentage = (1 - err / size(x, 1))*100

errors = abs(train(:,5) - y111');
err = sum(errors);
naive_percentage = (1 - err / size(x, 1))*100



size_data = size(data);
eta1 = .002;
eta2 = .003;
no_grp = 3;  
no_nodes = 4;
in =4;
itr = 200;

n0 = size_data(1);
n= size_data(1)/no_grp;
weights = [0,0.9,0.6,0.3];
for i=1:size_data(1)
    if(data(i,in+1)==0)
        data(i,in+1)=0.2;
    elseif(data(i,in+1)==1)
        data(i,in+1)=0.6;
    else
        data(i,in+1)=1;
    end
end
%r = randi(size_data(1),1,4);
r=[100,154,71,219];
c = data(r,1:4);
bias=0;
%r=randperm(150);
for i=1:itr%training
    for k=1:size_data(1)
        z1(k) = euclidean(data(k,1:4),c(1,:));%euclidean distance calculation.
        z2(k) = euclidean(data(k,1:4),c(2,:));
        z3(k) = euclidean(data(k,1:4),c(3,:));
        z4(k) = euclidean(data(k,1:4),c(4,:));
        phi1(k) = exp(-((z1(k))^2));
        phi2(k) = exp(-((z2(k))^2));
        phi3(k) = exp(-((z3(k))^2));
        phi4(k) = exp(-((z4(k))^2));
        phi = [phi1(k) phi2(k) phi3(k) phi4(k)];

        y(k)=(phi1(k)*weights(1,1))+(phi2(k)*weights(1,2))+(phi3(k)*weights(1,3))+(phi4(k)*weights(1,4));
        y(k)=y(k)+bias;
        for j=1:no_nodes
            c(j,:) = c(j,:) + eta1*(data(k,in+1) -y(k))*weights(j)*(phi(j)*2) *(data(k,1:4)-c(j,:));%centre updation
            weights(j) = weights(j) + eta2*(data(k,in+1) -y(k))*phi(j);%weight updation.
        end
        e(k) = data(k,in+1) -y(k);
    end
    err(i) = mse(e);%mean square error calculation
end

%figure;plot(err);title('Mean Square Error');xlabel('iteration --->');


y=[];
mismatch=0;
mismatch1=0;
for k=1:size_data(1)%testing
    z1(k) = euclidean(data((k),1:4),c(1,:));%euclidean distance calculation.
    z2(k) = euclidean(data((k),1:4),c(2,:));
    z3(k) = euclidean(data((k),1:4),c(3,:));
    z4(k) = euclidean(data((k),1:4),c(4,:));
    phi1(k) = exp(-((z1(k))^2));
    phi2(k) = exp(-((z2(k))^2));
    phi3(k) = exp(-((z3(k))^2));
    phi4(k) = exp(-((z4(k))^2));
    phi = [phi1(k) phi2(k) phi3(k) phi4(k)];
   y(k)=(phi1(k)*weights(1,1))+(phi2(k)*weights(1,2))+(phi3(k)*weights(1,3))+(phi4(k)*weights(1,4));
    y(k)=y(k)+bias;
    if(y(k)>=data(k,5)+.2 || y(k)<=data(k,5)-.2)
        mismatch1=mismatch1+1;
    end
end
%mismatch;
%for k=1:size_data(1)
%    if(y(k)<=0.3)
%        if(data(k,5)<=.3)
%        else
%            mismatch=mismatch+1;
%        end
%    elseif(y(k)>0.33&&y(k)<=0.9)
%        if(data(k,5)>.33&&data(k,5)<=.9)
%        else
%            mismatch=mismatch+1;
%        end
%    else
%        if(data(k,5)>1.5)
%        else
%            mismatch=mismatch+1;
%        end
%    end
%end

    



%RBF_percentage = (1 - mismatch/ size_data(1))*100
RBF_percentage1 = (1 - mismatch1/ size_data(1))*100












            