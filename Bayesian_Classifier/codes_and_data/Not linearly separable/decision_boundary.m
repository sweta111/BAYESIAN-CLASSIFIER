% -----for linearly seperable data-------
% data= load('D:\IIT\MS\sem 1\PR\Assignment2\Linearly Separable\12_ls.txt');
% training = cell(3,1);
% training{1}=data(1:500,:);
% training{2}=data(501:1000,:);
% training{3}=data(1001:1500,:);

%--------for non-linearly separable data-----------
% training{1}=load('class1.txt');
% training{2}=load('class2.txt');
% training{3}=load('class3.txt');
sample_means = cell(length(training),1);
 
for i=1:length(training),
    sample_means{i} = mean(training{i});
end

xrange = [-5 50];
yrange = [5 60];
inc = 0.04;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x); 
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); 
dist = [];
 
for i=1:length(training),
    disttemp = sum(abs(xy - repmat(sample_means{i}, [numxypairs 1])), 2);
    dist = [dist disttemp]; 
end

[m,idx] = min(dist, [], 2);
decisionmap = reshape(idx, image_size);
figure;
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [0.3 1 0.3; 0.3 0.3 1;1 0.3 0.3];
colormap(cmap);
plot(training{1}(:,1),training{1}(:,2), 'r.');
plot(training{2}(:,1),training{2}(:,2), 'go');
plot(training{3}(:,1),training{3}(:,2), 'b*');
title('Decision boundary and surface')
legend('Class 1', 'Class 2', 'Class 3')
xlabel('Dim 1');
ylabel('Dim 2');
a1=sample_means{1}
a2=sample_means{2}
a3=sample_means{3}
% l1=plot([a1(1) a2(1)], [a1(2) a2(2)],'LineWidth',2)
% l2=plot([a1(1) a3(1)], [a1(2) a3(2)],'LineWidth',2)
% l3=plot([a3(1) a2(1)], [a3(2) a2(2)],'LineWidth',2)
% 
% ----------gaussian pdf plot-----------
% x1 = 4:1:48; x2 = 4:1:48;
% [X1,X2] = meshgrid(x1,x2);
% F1 = mvnpdf([X1(:) X2(:)],sample_means{1}',cov1');
% F1 = reshape(F1,length(x2),length(x1));
% surf(x1,x2,F1);
% caxis([min(F1(:))-.5*range(F1(:)),max(F1(:))]);
% axis([4 48 4 48 0 0.0025])
% xlabel('x1'); ylabel('x2'); zlabel('Probability Density of class-1 for non-linear data');
% hold on;
% 
% F2 = mvnpdf([X1(:) X2(:)],mean2',cov2');
% F2 = reshape(F2,length(x2),length(x1));
% surf(x1,x2,F2);
% caxis([min(F2(:))-.5*range(F2(:)),max(F2(:))]);
% 
% F3 = mvnpdf([X1(:) X2(:)],mean3',cov3');
% F3 = reshape(F3,length(x2),length(x1));
% surf(x1,x2,F3);
% caxis([min(F3(:))-.5*range(F3(:)),max(F3(:))]);
