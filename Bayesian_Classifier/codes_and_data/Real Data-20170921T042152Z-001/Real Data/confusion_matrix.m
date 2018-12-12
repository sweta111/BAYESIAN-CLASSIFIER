case_no=5;
d=load('group_12.txt');
class1=d(1:500,:);
class2=d(501:1000,:);
class3=d(1001:end,:);

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
switch(case_no)
    case 1
    	conv=mean(cat(3,cov1,cov2,cov3),3);
		cov1=conv;cov2=conv;cov3=conv;
	case 2
	case 3
		conv=mean(cat(3,cov1,cov2,cov3),3);
		maxcov=max(conv(:));
		conv=maxcov * eye(size(conv));
		cov1=conv;cov2=conv;cov3=conv;
	case 4
		conv=mean(cat(3,cov1,cov2,cov3),3);
		conv = conv .* eye(size(conv));
		cov1=conv;cov2=conv;cov3=conv;
	case 5
		cov1 = cov1 .* eye(size(cov1));
		cov2 = cov2 .* eye(size(cov2));
		cov3 = cov3 .* eye(size(cov3));
end
cm=[1 0 0; 0 1 0; 0 0 1];
C =[repmat(cm(1,:),size(class1_train,1), 1);repmat(cm(2,:),size(class2_train,1),1);  repmat(cm(3,:), size(class3_train,1), 1)];
% % % % Discriminant function for each class
syms x1 x2
W_1=-0.5*inv(cov1);   w_1=inv(cov1)*mean1;
W_10=-(0.5*mean1'*inv(cov1)*mean1)-(0.5*log(det(cov1)))+log(1/3);
W_2=-0.5*inv(cov2);   w_2=inv(cov2)*mean2;
W_20=-(0.5*mean2'*inv(cov2)*mean2)-(0.5*log(det(cov2)))+log(1/3);
W_3=-0.5*inv(cov3);   w_3=inv(cov3)*mean3;
W_30=-(0.5*mean3'*inv(cov3)*mean3)-(0.5*log(det(cov3)))+log(1/3);
g1=[x1 x2] *W_1*[x1;x2]+ w_1'*[x1;x2] +  W_10;
g2=[x1 x2] *W_2*[x1;x2]+ w_2'*[x1;x2] +  W_20;
g3=[x1 x2] *W_3*[x1;x2]+ w_3'*[x1;x2] +  W_30;

% % % % Confusion matrix
x=vertcat(class1_test,class2_test,class3_test);
a1=zeros(size(class1_test));a1(:,1)=a1(:,1)+1;
a2=zeros(size(class2_test));a2(:,2)=a2(:,2)+1;
a3=zeros(size(class3_test));a3(:,3)=a3(:,3)+1;
gt=vertcat(a1,a2,a3);
clear a1 a2 a3;

x1=x(:,1);x2=x(:,2);
y=[subs(g1),subs(g2),subs(g3)];
maxm = max(y,[],2);
pred = bsxfun(@eq,y,maxm);
plotconfusion(gt',pred')