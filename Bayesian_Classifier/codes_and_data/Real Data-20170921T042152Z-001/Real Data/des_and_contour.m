clear;
case_no=5;
data= load('group_12.txt');
training = cell(3,1);
training{1}=data(1:500,:);
training{2}=data(501:1000,:);
training{3}=data(1001:1500,:);
 
for i=1:length(training),
    sample_means{i} = mean(training{i});
end
xrange = [100 3000];
yrange = [100 3000];
inc = 10;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x); 
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); 
dist = [];

%----------------------discriminant functions-----------------
class1=data(1:500,:);
class2=data(501:1000,:);
class3=data(1001:1500,:);

class1_train = sortrows(class1(randperm(size(class1,1)*0.7),:),1);
class2_train = sortrows(class2(randperm(size(class2,1)*0.7),:),1);
class3_train = sortrows(class3(randperm(size(class3,1)*0.7),:),1);
class1_test  = [class1(randperm(size(class1,1)*0.3),:),ones(size(class1,1)*0.3,1)];
class2_test  = [class2(randperm(size(class2,1)*0.3),:),ones(size(class2,1)*0.3,1)];
class3_test  = [class3(randperm(size(class3,1)*0.3),:),ones(size(class3,1)*0.3,1)];

data=[class1_train;class2_train;class3_train];

mean1= mean(class1)';
mean2= mean(class2)';
mean3= mean(class3)';
cov1=[cov(class1,1)]';
cov2=[cov(class2,1)]';
cov3=[cov(class3,1)]';
switch(case_no)
    case 1
    	cov=mean(cat(3,cov1,cov2,cov3),3);
		cov1=cov;cov2=cov;cov3=cov;
	case 2
		%do nothing
	case 3
		cov=mean(cat(3,cov1,cov2,cov3),3);
		maxcov=max(cov(:));
		cov=maxcov * eye(size(cov))
		cov1=cov;cov2=cov;cov3=cov;
	case 4
		cov=mean(cat(3,cov1,cov2,cov3),3);
		cov = cov.*eye(size(cov));
		cov1=cov;cov2=cov;cov3=cov;
	case 5
		cov1 = cov1 .* eye(size(cov1));
		cov2 = cov2 .* eye(size(cov2));
		cov3 = cov3 .* eye(size(cov3));
end
cm=[1 0 0; 0 1 0; 0 0 1];
C = [repmat(cm(1,:), size(class1_train,1), 1);  repmat(cm(2,:), size(class2_train,1), 1);  repmat(cm(3,:), size(class3_train,1), 1)];
syms x1 x2
W_1=[-0.5*inv(cov1)];
w_1=inv(cov1)*mean1;
W_10=-(0.5*mean1'*inv(cov1)*mean1)-(0.5*log(det(cov1)))+log(1/3);
W_2=[-0.5*inv(cov2)];
w_2=inv(cov2)*mean2;
W_20=-(0.5*mean2'*inv(cov2)*mean2)-(0.5*log(det(cov2)))+log(1/3);
W_3=[-0.5*inv(cov3)];
w_3=inv(cov3)*mean3;
W_30=-(0.5*mean3'*inv(cov3)*mean3)-(0.5*log(det(cov3)))+log(1/3);
g1=[x1 x2] *W_1*[x1;x2]+ w_1'*[x1;x2] +  W_10;
g2=[x1 x2] *W_2*[x1;x2]+ w_2'*[x1;x2] +  W_20;
g3=[x1 x2] *W_3*[x1;x2]+ w_3'*[x1;x2] +  W_30;
x1=xy(:,1);x2=xy(:,2);
val_g1=subs(g1);
val_g2=subs(g2);
val_g3=subs(g3);
dist=[val_g1 , val_g2, val_g3];

[m,idx] = max(dist, [], 2);
decisionmap = reshape(idx, image_size);
% figure;
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.7 0.7 ; 0.7 1 0.7;0.5 0.5 1];
colormap(cmap);
 
plot(class1_test(:,1),class1_test(:,2), 'rO'); hold on;
plot(class2_test(:,1),class2_test(:,2), 'mo'); hold on;
plot(class3_test(:,1),class3_test(:,2), 'g*'); hold on;

title('Decision boundary and surface');
legend('Class 1 test data', 'Class 2 test data', 'Class 3 test data');
xlabel('Feature 1');
ylabel('Feature 2');
hold on;
a1=sample_means{1};
a2=sample_means{2};
a3=sample_means{3};

x1 = -70:0.5:70; x2 = -70:0.5:70;
[X1,X2] = meshgrid(x1,x2);
F1 = mvnpdf([X1(:) X2(:)],mean1',cov1');
F1 = reshape(F1,length(x2),length(x1));
mvncdf([0 0],[1 1],mean1',cov1');

F2 = mvnpdf([X1(:) X2(:)],mean2',cov2');
F2 = reshape(F2,length(x2),length(x1));
mvncdf([0 0],[1 1],mean2',cov2');

F3 = mvnpdf([X1(:) X2(:)],mean3',cov3');
F3 = reshape(F3,length(x2),length(x1));
mvncdf([0 0],[1 1],mean3',cov3');

[C1,h1] =contour(x1,x2,F1,8);
set(h1,'LineWidth',2)
set(h1,'LineColor',[0 0 0.5])
hold on;
[C1,h1]=contour(x1,x2,F2,8);
set(h1,'LineWidth',2)
set(h1,'LineColor',[0.6 0.2 0.5])
hold on;
[C1,h1]=contour(x1,x2,F3,8);
set(h1,'LineWidth',2)
set(h1,'LineColor',[0.4 0.3 0.1])
hold on;
xlim([100,1500])