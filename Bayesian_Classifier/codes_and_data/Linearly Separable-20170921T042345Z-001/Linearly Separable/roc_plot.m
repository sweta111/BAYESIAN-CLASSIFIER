clear;
clc;
data=load('12_ls.txt');
class1=data(1:500,:);
class2=data(501:1000,:);
class3=data(1001:1500,:);
class1_train = sortrows(class1(randperm(size(class1,1)*0.7),:),1);
class2_train = sortrows(class2(randperm(size(class2,1)*0.7),:),1);
class3_train = sortrows(class3(randperm(size(class3,1)*0.7),:),1);
class1_test  = [class1(randperm(size(class1,1)*0.3),:),ones(size(class1,1)*0.3,1)];
class2_test  = [class2(randperm(size(class2,1)*0.3),:),ones(size(class2,1)*0.3,1)*2];
class3_test  = [class3(randperm(size(class3,1)*0.3),:),ones(size(class3,1)*0.3,1)*3];
data=[class1_train;class2_train;class3_train];

mean1= mean(class1_train)';
mean2= mean(class2_train)';
mean3= mean(class3_train)';
cov1=cov(class1_train)';
cov2=cov(class2_train)';
cov3=cov(class3_train)';

tp = zeros(24,5);
fp = zeros(24,5);
for i=1:5
    case_no=i;
switch(case_no)
    case 1
    	cov=mean(cat(3,cov1,cov2,cov3),3);
		cov1=cov;cov2=cov;cov3=cov;
	case 2
	case 3
		cov=mean(cat(3,cov1,cov2,cov3),3);
		maxcov=max(cov(:));
		cov=maxcov * eye(size(cov));
		cov1=cov;cov2=cov;cov3=cov;
	case 4
		cov=mean(cat(3,cov1,cov2,cov3),3);
		cov = cov .* eye(size(cov));
		cov1=cov;cov2=cov;cov3=cov;
	case 5
		cov1 = cov1 .* eye(size(cov1));
		cov2 = cov2 .* eye(size(cov2));
		cov3 = cov3 .* eye(size(cov3));
end
cm=[1 0 0; 0 1 0; 0 0 1];
C =[repmat(cm(1,:),size(class1_train,1), 1);repmat(cm(2,:),size(class2_train,1),1);  repmat(cm(3,:), size(class3_train,1), 1)];
syms x1 x2;
W_1=-0.5*inv(cov1);   w_1=inv(cov1)*mean1;
W_10=-(0.5*mean1'*inv(cov1)*mean1)-(0.5*log(det(cov1)))+log(1/3);
W_2=-0.5*inv(cov2);   w_2=inv(cov2)*mean2;
W_20=-(0.5*mean2'*inv(cov2)*mean2)-(0.5*log(det(cov2)))+log(1/3);
W_3=-0.5*inv(cov3);   w_3=inv(cov3)*mean3;
W_30=-(0.5*mean3'*inv(cov3)*mean3)-(0.5*log(det(cov3)))+log(1/3);
g1=[x1 x2] *W_1*[x1;x2]+ w_1'*[x1;x2] +  W_10;
g2=[x1 x2] *W_2*[x1;x2]+ w_2'*[x1;x2] +  W_20;
g3=[x1 x2] *W_3*[x1;x2]+ w_3'*[x1;x2] +  W_30;

% -----------------ROC plot-------------------

a1=zeros(size(class1_test));a1(:,1)=a1(:,1)+1;
a2=zeros(size(class2_test));a2(:,2)=a2(:,2)+1;
a3=zeros(size(class3_test));a3(:,3)=a3(:,3)+1;
gt=vertcat(a1,a2,a3);

% x1=x(:,1);x2=x(:,2);
% % % % for class 1
x1=class1(:,1);x2=class1(:,2);
y=[subs(g1),subs(g2),subs(g3)];
% y = y - min(y(:));
% y = y ./ (max(y(:))-min(y(:)));
target=y(:,1);
nontarget=[y(:,2),y(:,3)];

% % % % for class 2
x1=class2(:,1);x2=class2(:,2);
y=[subs(g1),subs(g2),subs(g3)];

target=[target;y(:,2)];
nontarget=[nontarget;y(:,1),y(:,3)];

x1=class3(:,1);x2=class3(:,2);
y=[subs(g1),subs(g2),subs(g3)];

target=[target;y(:,3)];
nontarget=[nontarget;y(:,1),y(:,2)];
% thresholds=[-80 -70 -60 -50 -40 -30 -25 -20 -15 -10 -5 ];
tpr=0;fpr=0;
for thre=-120:+5:-5
        tpr=[tpr;sum(target>thre)/(size(target,1))];
        fpr=[fpr;sum(sum(nontarget>thre))/(size(nontarget,1)*2)];
end
fpr = fpr - min(fpr(:));
fpr = fpr ./ (max(fpr(:))- min(fpr(:)));
tpr = tpr - min(tpr(:));
tpr = tpr ./ (max(tpr(:))- min(tpr(:)));
tp(:,i)=tpr(2:end); fp(:,i)=fpr(2:end);
% plot(fpr(2:end),tpr(2:end),'-o');
end
x=[0 :0.1:1];y=x;
h0=plot(x,y,'-'); set(h0,'LineStyle',' - -');hold on;
x = [0.6 0.5];
y = [0.4 0.5];
annotation('textarrow',x,y,'String','Random guess')
h1=plot(fp(:,1),tp(:,1),'r'); set(h1,'LineWidth',15);hold on;
h2=plot(fp(:,2),tp(:,2),'g');set(h2,'LineWidth',10);hold on;
h3=plot(fp(:,3),tp(:,3),'b');set(h3,'LineWidth',5);hold on;
h4=plot(fp(:,4),tp(:,4),'m');set(h4,'LineWidth',3);hold on;
h5=plot(fp(:,5),tp(:,5),'c');set(h5,'LineWidth',1);
title('ROC curve for linear data');
legend('Random guess','Case 1','Case 2','Case 3','Case 4','Case 5');
xlabel('False positive rate');
ylabel('True positive rate');

    