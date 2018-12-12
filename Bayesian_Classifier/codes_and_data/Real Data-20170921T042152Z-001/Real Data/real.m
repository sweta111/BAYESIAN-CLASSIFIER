clear;
file=load('group_12.txt');

C1=file(1:500,:);
C1_tr = C1(randperm(size(C1,1)*0.7),:);
C1_ts = C1(randperm(size(C1,1)*0.3),:);

C2=file(501:1000,:);
C2_tr = C2(randperm(size(C2,1)*0.7),:);
C2_ts = C2(randperm(size(C2,1)*0.3),:);

C3=file(1001:1500,:);
C3_tr = C3(randperm(size(C3,1)*0.7),:);
C3_ts = C3(randperm(size(C3,1)*0.3),:);

M1=[mean(C1_tr(:,1)); mean(C1_tr(:,2))];
M2=[mean(C2_tr(:,1)); mean(C2_tr(:,2))];
M3=[mean(C3_tr(:,1)); mean(C3_tr(:,2))];

Co1=[cov(C1_tr)]';
Co2=[cov(C2_tr)]';
Co3=[cov(C3_tr)]';
%--------------------covariance for each case-----------------%
% if caseno==1

%     Co1=([cov(C1_tr)]'+[cov(C2_tr)]'+[cov(C3_tr)]')/3;  
%     Co2=Co1;
%     Co3=Co1;
% elseif caseno==2
%      Co1=[cov(C1_tr)]';
%      Co2=[cov(C2_tr)]';
%      Co3=[cov(C3_tr)]';
% else if caseno==3
%      Coo1=[diag(var(C1_tr))]';  
%      Coo2=[diag(var(C2_tr))]';
%      Coo3=[diag(var(C3_tr))]'; 
%      Co1=(Coo1+Coo2+Coo3)/3;
%      Co2=Co1;
%      Co3=Co1;
%     elseif caseno==4
 
%        Coo1=diag([diag(cov(C1_tr))]');
%        Coo2=diag([diag(cov(C2_tr))]');
%        Coo3=diag([diag(cov(C3_tr))]'); 
%        Co1=(Coo1+Coo2+Coo3)/3;
%        Co2=Co1;
%        Co3=Co1;
%     elseif caseno==5
%          Co1=diag([diag(cov(C1_tr))]');
%          Co2=diag([diag(cov(C2_tr))]');
%          Co3=diag([diag(cov(C3_tr))]');  
% end 

W1=[-0.5*inv(Co1)];
w1=inv(Co1)*M1;
w10=-(0.5*M1'*inv(Co1)*M1)-(0.5*log(det(Co1)))+log(350/1050);

W2=[-0.5*inv(Co2)];
w2=inv(Co2)*M2;
w20=-(0.5*M2'*inv(Co2)*M2)-(0.5*log(det(Co2)))+log(350/1050);

W3=[-0.5*inv(Co3)];
w3=inv(Co3)*M3;
w30=-(0.5*M3'*inv(Co3)*M3)-(0.5*log(det(Co3)))+log(350/1050);

x=sym('x',[2 1]);
xt=sym('x',[1 2]);

g1=(xt*W1*x)+(w1'*x)+w10;
g2=(xt*W2*x)+(w2'*x)+w20;
g3=(xt*W3*x)+(w3'*x)+w30;

cm=[1 0 0; 0 1 0; 0 0 1];
C = [repmat(cm(1,:), size(C1_ts,1), 1);  repmat(cm(2,:), size(C2_ts,1), 1);  repmat(cm(3,:), size(C3_ts,1), 1)];
C_ts=[C1_ts;C2_ts;C3_ts];

figure(1)
gscatter(file(:,1),file(:,2),C,'rgb','oxd');
hold on;
legend('Class-1','Class-2','Class-3');

set(ezplot(g1-g2,[0 3000]),'color',[1 0 0]);
set(ezplot(g2-g3,[0 3000]),'color',[0 0 1]);
set(ezplot(g1-g3,[0 3000]),'color',[0 1 0]);

% % --------------------plotting pdf and contour-----------------------------%
% x1 = 4:1:48; x2 = 4:1:48;
% [X1,X2] = meshgrid(x1,x2);
% F1 = mvnpdf([X1(:) X2(:)],M1',Co1');
% F1 = reshape(F1,length(x2),length(x1));
% F2 = mvnpdf([X1(:) X2(:)],M2',Co2');
% F2 = reshape(F2,length(x2),length(x1));
% F3 = mvnpdf([X1(:) X2(:)],M3',Co3');
% F3 = reshape(F3,length(x2),length(x1));
% 
%  figure(3)
% %  cm=[1 0 0; 0 1 0; 0 0 1];
% %  C = [repmat(cm(1,:), size(C1,1), 1);  repmat(cm(2,:), size(C2,1), 1);  repmat(cm(3,:), size(C3,1), 1)];
% %  gscatter(file(:,1),file(:,2),C,'rgb','odx');
% %  hold on;
% h=axes;
% contourf(x1,x2,F1,10,'edgecolor','white');
% axis([120 2600 120 2600  0.035]);
% hold on;
% surf(x1,x2,F1+0.02);
% 
% contourf(x1,x2,F2,10,'edgecolor','white');
% surf(x1,x2,F2+0.02);
% 
% contourf(x1,x2,F3,10,'edgecolor','white');
% surf(x1,x2,F3+0.02);
% 
% set(gca,'fontsize',14);
% set(h,'Box','off');
% xlabel('feature-1'); ylabel('feature-2');zlabel('Probability density function');
% title('Constant denstity function and contour');
% 
% %-------------------------plotting eigen vectors----------------------%
% Cov21=[cov(C1)]';
% Cov22=[cov(C2)]';
% Cov23=[cov(C3)]';
% [V21 D]=eig(Cov21);
% [V22 D]=eig(Cov22);
% [V23 D]=eig(Cov23);
% contour(x1,x2,F1,5,'LineWidth',3);
% %caxis([min(F1(:))-.5*range(F1(:)),max(F1(:))]);
% axis([100 3000 200 3000]);
% hold on;
% contour(x1,x2,F2,5,'LineWidth',3);
% %caxis([min(F2(:))-.5*range(F2(:)),max(F2(:))]);
% contour(x1,x2,F3,5,'LineWidth',3);
% colormap(hot);
% %caxis([min(F3(:))-.5*range(F3(:)),max(F3(:))]);
% set(plot([M1(1,:) 15*V21(1,1)],[M1(2,:) 15*V21(2,1)],'r'),'LineWidth',5);
% set(plot([M1(1,:) 15*V21(1,2)],[M1(2,:) 15*V21(2,2)],'g'),'LineWidth',5);
% set(plot([M2(1,:) 15*V22(1,1)],[M2(2,:) 15*V22(2,1)],'b'),'LineWidth',5);
% set(plot([M2(1,:) 40*V22(1,2)],[M2(2,:) 60*V22(2,2)],'c'),'LineWidth',5);
% set(plot([M3(1,:) 15*V23(1,1)],[M3(2,:) 15*V23(2,1)],'m'),'LineWidth',5);
% set(plot([M3(1,:) 15*V23(1,2)],[M3(2,:) 15*V23(2,2)],'y'),'LineWidth',5);
% set(gca,'fontsize',15);
% xlabel('Dim-1');
% ylabel('Dim-2');
% title('Constant density curves and eigen vectors');
% %---------------------------------------------------------------------%

%--------------prediction of classes---------------------------%

C11predicted=zeros(size(C1_ts,1),3);
C21predicted=zeros(size(C2_ts,1),3);
C31predicted=zeros(size(C3_ts,1),3);

for i=1:size(C1_ts,1)
x11=C1_ts(i,1);
x12=C1_ts(i,2);
x1=[x11;x12];
x1t=x1';
g1(i)=(x1t*W1*x1)+(w1'*x1)+w10;
g2(i)=(x1t*W2*x1)+(w2'*x1)+w20;
g3(i)=(x1t*W3*x1)+(w3'*x1)+w30;
if g1(i)>g2(i)
    if g1(i)>g3(i)
        C11predicted(i,1)=1;
    else
        C11predicted(i,3)=1;
    end
end
if g2(i)>g3(i)
    C11predicted(i,2)=1;
else
    C11predicted(i,3)=1; 
end
end

%class-2 prediction
for i=1:size(C2_ts)
x21=C2_ts(i,1);
x22=C2_ts(i,2);
x2=[x21;x22];
x2t=x2';
g1(i)=(x2t*W1*x2)+(w1'*x2)+w10;
g2(i)=(x2t*W2*x2)+(w2'*x2)+w20;
g3(i)=(x2t*W3*x2)+(w3'*x2)+w30;
if g1(i)>g2(i)
    if g1(i)>g3(i)
        C21predicted(i,1)=1;
    else
        C21predicted(i,3)=1;
    end
end
if g2(i)>g3(i)
    C21predicted(i,2)=1;
else
    C21predicted(i,3)=1; 
end
end

%class_3 prediction

for i=1:size(C3_ts)
x31=C3_ts(i,1);
x32=C3_ts(i,2);
x3=[x31;x32];
x3t=x3';
g1(i)=(x3t*W1*x3)+(w1'*x3)+w10;
g2(i)=(x3t*W2*x3)+(w2'*x3)+w20;
g3(i)=(x3t*W3*x3)+(w3'*x3)+w30;
if g1(i)>g2(i)
    if g1(i)>g3(i)
        C31predicted(i,1)=1;
    else
        C31predicted(i,3)=1;
    end
end
if g2(i)>g3(i)
    C31predicted(i,2)=1;
else
    C31predicted(i,3)=1; 
end
end

pred = [C11predicted ; C21predicted ; C31predicted];
target=zeros(size(C));
target(1:length(C1_ts),1) = 1;
target(length(C1_ts) + 1 : (2*length(C1_ts)) ,2) = 1;
target((2*length(C1_ts)) + 1 : (3*length(C1_ts)) , 3) = 1;
%-------------------------ploting confusion matrix----------------------%
plotconfusion(target',pred');
set(gca,'fontsize',18);
title('Confusion matrix for case2');
