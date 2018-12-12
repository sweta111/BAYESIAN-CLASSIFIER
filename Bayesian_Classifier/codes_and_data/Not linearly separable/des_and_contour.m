clear;
% -----for linearly seperable data-------
% data= load('D:\IIT\MS\sem 1\PR\Assignment2\Linearly Separable\12_ls.txt');
% training = cell(3,1);
% training{1}=data(1:500,:);
% training{2}=data(501:1000,:);
% training{3}=data(1001:1500,:);

% --------for non-linearly separable data-----------
training{1}=load('class1.txt');
training{2}=load('class2.txt');
training{3}=load('class3.txt');
sample_means = cell(length(training),1);
 
for i=1:length(training),
    sample_means{i} = mean(training{i});
end
% 
xrange = [0 45];
yrange = [5 55];
% xrange = [-5 20];
% yrange = [-5 20];
inc = 0.04;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x); 
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); 
dist = [];

%----------------------discriminant functions-----------------
case_no=5
% class1=data(1:500,:);
% class2=data(501:1000,:);
% class3=data(1001:1500,:);
class1=load('class1.txt');
class2=load('class2.txt');
class3=load('class3.txt');

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
cov1=cov(class1,1)';
cov2=cov(class2,1)';
cov3=cov(class3,1)';
switch(case_no)
    case 1
    	cov=mean(cat(3,cov1,cov2,cov3),3);
		cov1=cov;cov2=cov;cov3=cov;
	case 2
		%do nothing
	case 3
		cov=mean(cat(3,cov1,cov2,cov3),3);
		maxcov=max(cov(:));
		cov=maxcov * eye(size(cov));
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
W_1=-0.5*inv(cov1);
w_1=inv(cov1)*mean1;
W_10=-(0.5*mean1'*inv(cov1)*mean1)-(0.5*log(det(cov1)))+log(1/3);
W_2=-0.5*inv(cov2);
w_2=inv(cov2)*mean2;
W_20=-(0.5*mean2'*inv(cov2)*mean2)-(0.5*log(det(cov2)))+log(1/3);
W_3=-0.5*inv(cov3);
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

%-------------------Drawing decisionmap---------------------------
[m,idx] = max(dist, [], 2);
decisionmap = reshape(idx, image_size);
% figure;
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.7 0.7 ; 0.7 1 0.7;0.5 0.5 1];
colormap(cmap);
% Training data for scatter plot-------------------------
% plot(training{1}(:,1),training{1}(:,2), 'rO'); hold on;
% plot(training{2}(:,1),training{2}(:,2), 'mo'); hold on;
% plot(training{3}(:,1),training{3}(:,2), 'g*'); hold on;
% -------------------------------------------------------
% 
plot(class1_test(:,1),class1_test(:,2), 'rO'); hold on;
plot(class2_test(:,1),class2_test(:,2), 'mo'); hold on;
plot(class3_test(:,1),class3_test(:,2), 'g*'); hold on;

% title('Decision boundary and surface');
legend('Class 1 test data', 'Class 2 test data', 'Class 3 test data');
% legend('Class 1', 'Class 2', 'Class 3');
xlabel('Feature 1');
ylabel('Feature 2');
hold on;
a1=sample_means{1};
a2=sample_means{2};
a3=sample_means{3};
% l1=plot([a1(1) a2(1)], [a1(2) a2(2)],'LineWidth',2);
% l2=plot([a1(1) a3(1)], [a1(2) a3(2)],'LineWidth',2);
% l3=plot([a3(1) a2(1)], [a3(2) a2(2)],'LineWidth',2);

% ------------------contour plot---------------------
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

% [C1,h1] =contour(x1,x2,F1,[.00001 .0001 .001 .01 .05 :.1:.95 .99 .999 .9999 1]);
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

% -----------------gaussian pdf plot-----------------
% x1 = 4:1:48; x2 = 4:1:48;
% x1 = -1.5:0.05:7.2; x2 = -1.5:0.05:6.8;
% [X1,X2] = meshgrid(x1,x2);
% F1 = mvnpdf([X1(:) X2(:)],sample_means{1},cov1);
% F1 = reshape(F1,length(x2),length(x1));
% hSurface=surf(x1,x2,F1); hold on;

% rotate(hSurface,[1 0 0],25)
% caxis([0.5,0.9]);
% % axis([4 48 4 48 0 0.0025])
% % xlabel('x1'); ylabel('x2'); zlabel('Probability Density of class-1 for non-linear data');
% hold on;
% x1 = 9.8:0.05:18.9; x2 =  4.8:0.05:14.9;
% F2 = mvnpdf([X1(:) X2(:)],mean2',cov2');
% F2 = reshape(F2,length(x2),length(x1));
% surf(x1,x2,F2); hold on;
% caxis([min(F2(:))-.5*range(F2(:)),max(F2(:))]);
% 
% F3 = mvnpdf([X1(:) X2(:)],mean3',cov3');
% F3 = reshape(F3,length(x2),length(x1));
% surf(x1,x2,F3);
% caxis([min(F3(:))-.5*range(F3(:)),max(F3(:))]);
% rotate3d on;