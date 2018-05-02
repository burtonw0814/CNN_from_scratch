clear all;
close all;
clc;

% A convolutional neural network implemented in MATLAB from scratch
% The code parses HTML code from Google search results to bring in images
% of either Batman or Spiderman. The data is split into training,
% validation, and testing data before being trained. 

% Relevant literature:

% Adam Optimizer: The optimization algorithm utilized to train the network
% (I found that this works better than stochastic gradient descent)
% Kingma DP, Ba J. Adam: A method for stochastic optimization. 
% arXiv preprint arXiv:1412.6980. 2014 Dec 22.

% LeNet: Regarded as the eariest implementation of the modern CNN
% LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to document recognition. 
% Proceedings of the IEEE. 1998 Nov;86(11):2278-324.

% and of course, AlexNet: 
% Krizhevsky A, Sutskever I, Hinton GE. Imagenet classification with deep convolutional neural networks. 
% InAdvances in neural information processing systems 2012 (pp. 1097-1105).


% Get data
% Function looks for files in current directory
total_dat=get_data(); % Parses files with HTML code for image source URLs

% View example
picsize=128;
temp=imread(total_dat{5,1}); % Read image from  source
X_initial=imresize(temp,[picsize, picsize]); % Resize image to be consistent dimensions
X=double(X_initial);
X=reshape(X,[],1);

% % % Visualize imported data
  for iiii=1:10:size(total_dat,1)
      IDX=iiii;
      temp=imread(total_dat{IDX,1}); % Read image f

      if size(temp,3)==3
     
            figure(iiii)
            imshow(temp,[]);
            truesize([500 500]);
              if total_dat{IDX,2}==0
                 class='Batman';
              else
                  class='Spiderman';
              end
          
            title({['Class = ' class]});
            pause(0.5)
            close all
            
      end
     
 end

%%%%%%%%%%%%%%%%%%%% Define parameters %%%%%%%%%%%%%%%%%%%%%%

% Set up train, val, and test data
train_portion=0.6; train_cap=round(train_portion*size(total_dat,1));
val_portion=0.15; val_cap=train_cap+round(val_portion*size(total_dat,1));
test_portion=0.25;

train_dat=total_dat(1:train_cap,:);
val_dat=total_dat(train_cap+1:val_cap,:);
test_dat=total_dat(val_cap+1:end,:);

disp(['Size of training data = ' num2str(size(train_dat,1))])
disp(['Size of validation data = ' num2str(size(val_dat,1))])
disp(['Size of testing data = ' num2str(size(test_dat,1))])

% convolutional layer #1 parameters
stvs4conv=0.01;
input_size=[picsize picsize 3];
k1=[]; b1=[]; num_f1=8; s=5;
for j=1:num_f1
   k1{j}=stvs4conv*randn(s, s, input_size(3));
   b1{j}=stvs4conv;
   
    % Define first and second moment terms for optimization
    V_k1{j}=zeros(size(k1{j})); V_k1_cc{j}=zeros(size(k1{j}));
    S_k1{j}=zeros(size(k1{j})); S_k1_cc{j}=zeros(size(k1{j}));
    
    V_b1{j}=zeros(size(b1{j})); V_b1_cc{j}=zeros(size(b1{j}));
    S_b1{j}=zeros(size(b1{j})); S_b1_cc{j}=zeros(size(b1{j}));
end
k1_size=size(k1{1});

% Conv Layer #2 parameters
k2=[]; b2=[]; num_f2=16; 
for j=1:num_f2
   k2{j}=stvs4conv*randn(s, s , length(k1));
   b2{j}=stvs4conv;
   
    %  Define first and second moment terms 
    V_k2{j}=zeros(size(k2{j})); V_k2_cc{j}=zeros(size(k2{j}));
    S_k2{j}=zeros(size(k2{j})); S_k2_cc{j}=zeros(size(k2{j}));
    
    V_b2{j}=zeros(size(b2{j})); V_b2_cc{j}=zeros(size(b2{j}));
    S_b2{j}=zeros(size(b2{j})); S_b2_cc{j}=zeros(size(b2{j}));
    
end
k2_size=size(k2{1});

% Conv Layer #3 parameters
k3=[]; b3=[]; num_f3=32;
for j=1:num_f3
   k3{j}=stvs4conv*randn(s, s, length(k2));
   b3{j}=stvs4conv;
   
    %  Define first and second moment terms 
    V_k3{j}=zeros(size(k3{j})); V_k3_cc{j}=zeros(size(k3{j}));
    S_k3{j}=zeros(size(k3{j})); S_k3_cc{j}=zeros(size(k3{j}));
    
    V_b3{j}=zeros(size(b3{j})); V_b3_cc{j}=zeros(size(b3{j}));
    S_b3{j}=zeros(size(b3{j})); S_b3_cc{j}=zeros(size(b3{j}));
end
k3_size=size(k3{1});

stvs=0.01;
% FC Layer 1 parameters
num_pools=3;
num_nodes1=100;   
w1_f=stvs*randn(num_nodes1,(picsize/(2^num_pools))*(picsize/(2^num_pools))*num_f3);
b1_f=stvs*ones(num_nodes1,1);

%  Define first and second moment terms 
V_w1_f=zeros(size(w1_f)); V_w1_f_cc=zeros(size(w1_f));
S_w1_f=zeros(size(w1_f)); S_w1_f_cc=zeros(size(w1_f));
V_b1_f=zeros(size(b1_f)); V_b1_f_cc=zeros(size(b1_f));
S_b1_f=zeros(size(b1_f)); S_b1_f_cc=zeros(size(b1_f));

% FC Layer 2 parameters
num_nodes2=2;   
w2_f=stvs*randn(num_nodes2,num_nodes1);
b2_f=stvs*ones(num_nodes2,1);

%  Define first and second moment terms 
V_w2_f=zeros(size(w2_f)); V_w2_f_cc=zeros(size(w2_f));
S_w2_f=zeros(size(w2_f)); S_w2_f_cc=zeros(size(w2_f));
V_b2_f=zeros(size(b2_f)); V_b2_f_cc=zeros(size(b2_f));
S_b2_f=zeros(size(b2_f)); S_b2_f_cc=zeros(size(b2_f));

% Hyper parameters
num_epochs=5;
learning_rate=0.0001;
eta=learning_rate;
loss_vec=[]; p_vec=[];

% Adam hyperparameters
beta1=0.9;
beta2=0.999;
eps=0.00000001;
tot_ct=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two vectors that keep track of accuracy at the end of each epoch
train_accuracy=[]; val_accuracy=[]; 
for jjjj=1:num_epochs
    
    disp(['Epoch ' num2str(jjjj)])
    train_dat=shuffle_data(train_dat); % Shuffle training data at beginning of epoch
    
    for iiii=1:size(train_dat,1)
        
        %%%%%%%%%%% Get data for training iteration %%%%%%%%
        
            TRAIN_IDX=iiii;
            temp=imread(train_dat{TRAIN_IDX,1}); % Read image from source
             if length(size(temp))==3 % Make sure imported picture is color (depth of 3)
                X_initial=imresize(temp,[picsize, picsize]); % Resize image to be consistent dimensions for classification architecture
                X=double(X_initial);

                % Define ground truth vector in one-hot form
                if (train_dat{TRAIN_IDX,2})==0
                   Y=[1; 0]; 
                else
                   Y=[0; 1];
                end
                
                %%%%%%%%%%%%%%%%%%%%%% Forward pass %%%%%%%%%%%%%%%%%%%

                % pad1
                a=zeros(input_size(1)+4, input_size(2)+4, input_size(3),'double');
                a(3:end-2,3:end-2,1:3)=X;

                % conv1
                b=[]; c=[];
                for rr=1:num_f1 % Number of feature maps in first layer
                    for jj=1:size(a,2)-k1_size(1)+1 % Rows
                        for ii=1:size(a,1)-k1_size(2)+1 % Cols
                            kernel=reshape(k1{rr},1,[]);
                            temp=reshape(a(ii:ii+k1_size(1)-1,jj:jj+k1_size(2)-1,:),[],1);
                            b(ii,jj,rr)=kernel*temp;
                            c(ii,jj,rr)=b(ii,jj,rr)+b1{rr};
                        end
                    end
                end

                % relu1
                logical_matrix1=(c>=0);
                d=c.*logical_matrix1;

                % pool1
                e=[]; temp_ind1i=[]; temp_ind1j=[];
                jj=1;
                for jjj=0:2:size(d,2)-2
                    ii=1;
                    for iii=0:2:size(d,1)-2
                        temp=reshape(d(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f1);
                        [mx_temp, idx_temp] = max(temp);
                        temp_ind1j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj; % Store indices of max pool values for backprop
                        temp_ind1i(ii,jj,:)=iii+2-mod(idx_temp,2); 
                        e(ii,jj,:)=mx_temp;
                        ii=ii+1;
                    end
                    jj=jj+1;
                end
                
                % pad2
                f=zeros(size(e,1)+4, size(e,2)+4, size(e,3),'double');
                f(3:end-2, 3:end-2, 1:size(e,3))=e;

                % conv2
                g=[]; h=[];
                for rr=1:num_f2 % Number of feature maps in first layer
                    for jj=1:size(f,2)-k2_size(1)+1 % Rows
                        for ii=1:size(f,1)-k2_size(2)+1 % Cols
                            kernel=reshape(k2{rr},1,[]);
                            temp=reshape(f(ii:ii+k2_size(1)-1,jj:jj+k2_size(2)-1,:),[],1);
                            g(ii,jj,rr)=kernel*temp;
                            h(ii,jj,rr)=g(ii,jj,rr)+b2{rr};
                        end
                    end
                end

                % relu2
                logical_matrix2=(h>=0);
                i=h.*logical_matrix2 ;

                % pool2
                j=[]; temp_ind2i=[]; temp_ind2j=[];
                jj=1;
                for jjj=0:2:size(i,2)-2
                    ii=1;
                    for iii=0:2:size(i,1)-2
                         temp=reshape(i(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f2); 
                        [mx_temp, idx_temp] = max(temp);
                        temp_ind2j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj;
                        temp_ind2i(ii,jj,:)=iii+2-mod(idx_temp,2);
                        j(ii,jj,:)=mx_temp;
                        ii=ii+1;
                    end
                    jj=jj+1;
                end
                
                % Pad3
                k=zeros(size(j,1)+4, size(j,2)+4, size(j,3),'double');
                k(3:end-2,3:end-2,:)=j;
                
                % conv3
                l=[]; m=[];
                for rr=1:num_f3 % Number of feature maps
                    for jj=1:size(k,2)-k3_size(1)+1 % Rows
                        for ii=1:size(k,1)-k3_size(2)+1 % Cols
                            kernel=reshape(k3{rr},1,[]);
                            temp=reshape(k(ii:ii+k3_size(1)-1,jj:jj+k3_size(2)-1,:),[],1);
                            l(ii,jj,rr)=kernel*temp;
                            m(ii,jj,rr)=l(ii,jj,rr)+b3{rr};
                        end
                    end
                end
                
                % relu3
                logical_matrix3=(m>=0);
                n=m.*logical_matrix3;
                
                % pool3
                o=[]; temp_ind3i=[]; temp_ind3j=[];
                jj=1;
                for jjj=0:2:size(n,2)-2
                    ii=1;
                    for iii=0:2:size(n,1)-2
                        % Rearrange each feature map to column major vectors
                        temp=reshape(n(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f3);
                        [mx_temp, idx_temp] = max(temp);
                        temp_ind3j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj;
                        temp_ind3i(ii,jj,:)=iii+2-mod(idx_temp,2);
                        o(ii,jj,:)=mx_temp;
                        ii=ii+1;
                        
                    end
                    jj=jj+1;
                end

                % Vectorize for fully connected layer
                p=[]; fc=1;
                for kk=1:size(o,3)
                    for jj=1:size(o,2)
                        for ii=1:size(o,1)
                            p(fc,1)=o(ii,jj,kk); fc=fc+1;
                        end
                    end
                end
                

                % FC Layer 1
                q=w1_f*p; % Multiply by weight matrix
                r=q+b1_f; % Add a bias
                logical_matrix4=(r>=0); % relu
                s=r.*logical_matrix4;

                % FC Layer 2
                t=w2_f*s; % Multiply by weight matrix
                u=t+b2_f; % Add a bias
                v=exp(u)./sum(exp(u)); % Nonlinear activation using softmax

                % Loss layer
                loss = mean(-1*(1.*Y.*log(v)+(1-Y).*log(1-v))); 
                
                if mod(iiii,100)==0
                    disp(['Epoch ' num2str(jjjj) ' Step ' num2str(iiii) ' Loss = ' num2str(loss)]);
                    disp(['Prediction is ' num2str(round(v(1)))]);
                end
                
                loss_vec=[loss_vec; loss]; % Keep track of loss
                p_vec=[p_vec; round(v(1))]; % Keep track of predictions
                
                
                %%%%%%%%%%%%%%%%%%%% Begin backward pass %%%%%%%%%%%%%%%%%%%% 
                
                
                dldv=0.5*((v-Y)./(v-(v.*v)));

                dvdu=[]; % Derivative of softmax outputs w.r.t. softmax inputs
                for ii=1:length(v) % i = output --> softmax
                    for jj=1:length(v) % j = input --> dense2

                        if ii==jj
                            dvdu(ii,jj) = v(ii) * (1-v(jj));
                        else
                            dvdu(ii,jj) = -1 * v(ii) * v(jj);
                        end

                    end
                end
                dldu=dvdu*dldv;

                dldb2=dldu; % <-- Gradient of bias of FC layer 2
                dldt=dldu;

                dtdw2=s';
                dldw2=dldt*dtdw2; % <-- Gradient of weight matrix of FC layer 2

                dtds=w2_f';
                dlds=dtds*dldt;

                % Back prop through fully-connected ReLu
                dldr=zeros(size(r));
                dldr=logical_matrix4.*dlds;

                dldb1=dldr; % <-- Gradient of bias of FC layer 1
                dldq=dldr;

                dqdw1=p';
                dldw1=dldq*dqdw1; % <-- Gradient of weight matrix of FC layer 1
               
                dqdp=w1_f';
                dldp=dqdp*dldq;
                
                % Back prop through conv portion, starting with
                % unflattening the gradients back into grid format

                % Flatten back prop (ie unflatten)
                dldo=[]; fc=1;
                for kk=1:size(o,3)
                    for jj=1:size(o,2)
                        for ii=1:size(o,1)
                            dldo(ii,jj,kk)=dldp(fc); fc=fc+1;
                        end
                    end
                end
                
                
                % Pool 3 --> Use temp_ind3j/i to assign gradient elements
                dldn=zeros(size(n));
                for ii=1:size(temp_ind3i,1)
                    for jj=1:size(temp_ind3i,2)
                        for kk=1:size(temp_ind3i,3)
                            dldn(temp_ind3i(ii,jj,kk),temp_ind3j(ii,jj,kk),kk)=dldo(ii,jj,kk);
                        end
                        
                    end
                end
                
                % Relu 3
                dldm=zeros(size(m));
                dldm=logical_matrix3.*dldn;
                
                % Bias 3
                for ii=1:length(b3) 
                    dldb3_c{ii}=sum(sum(dldm(:,:,ii))); % Gradient of conv bias 3
                end
                
                dldl=dldm;
                
                % Conv 3
                dldk3_c=[]; % Gradient of conv filter 3
                dldk=zeros(size(k)); % Gradient of input to conv filter 3
                for rr=1:num_f3
                    dldk3_c{rr}=zeros(size(k3{rr}));
                    
                    for ii=1:size(l,1)
                        for jj=1:size(l,2)
                            
                            dldk3_c{rr} = dldk3_c{rr} + (k(ii:ii-1+k3_size(1), jj:jj-1+k3_size(2),:) * dldl(ii,jj,rr));
                            
                            dldk(ii:ii-1+k3_size(1), jj:jj-1+k3_size(2),:) =  dldk(ii:ii-1+k3_size(1), jj:jj-1+k3_size(2),:) + ... 
                                   k3{rr} *  dldl(ii,jj,rr);
                            
                        end
                    end
                end
                
                dldj=dldk;
                
                % Pool 2 --> Use temp_ind2j/i to assign gradient elements
                dldi=zeros(size(i));
                for ii=1:size(temp_ind2i,1)
                    for jj=1:size(temp_ind2i,2)
                        for kk=1:size(temp_ind2i,3)
                           dldi(temp_ind2i(ii,jj,kk),temp_ind2j(ii,jj,kk),kk)=dldj(ii,jj,kk);
                        end 
                    end
                end
                
                % Through ReLu 2
                dldh=zeros(size(h));
                dldh=logical_matrix2.*dldi;
                
                % Through conv bias 2
                for ii=1:length(b2)
                   dldb2_c{ii}=sum(sum(dldh(:,:,ii)));  % Gradient of conv bias 2
                end
                
                dldg=dldh;
                
                % Through conv kernels 2
                dldk2_c=[];  % Gradient of conv filter 2
                dldf=zeros(size(f)); % Gradient of input to conv filter 2
                for rr=1:num_f2
                    dldk2_c{rr}=zeros(size(k2{rr}));
                    
                    for ii=1:size(g,1)
                        for jj=1:size(g,2)
                            
                            dldk2_c{rr} = dldk2_c{rr} + (f(ii:ii-1+k2_size(1), jj:jj-1+k2_size(2),:) * dldg(ii,jj,rr));
                            
                            dldf(ii:ii-1+k2_size(1), jj:jj-1+k2_size(2),:) =  dldf(ii:ii-1+k2_size(1), jj:jj-1+k2_size(2),:) + ... 
                                   k2{rr} *  dldg(ii,jj,rr);
                            
                        end
                    end
                end
                
                dlde=dldf; % Pad
                  
                % Through pool 2 --> Use temp_ind1j/i to assign gradient elements
                dldd=zeros(size(d));
                for ii=1:size(temp_ind1i,1)
                    for jj=1:size(temp_ind1i,2)
                        for kk=1:size(temp_ind1i,3)
                           dldd(temp_ind1i(ii,jj,kk),temp_ind1j(ii,jj,kk))=dlde(ii,jj,kk);
                        end 
                    end
                end
                
                % Through ReLu 1
                dldc=zeros(size(c));
                dldc=logical_matrix1.*dldd;
                
                % Through conv bias 1
                for ii=1:length(b1)
                   dldb1_c{ii}=sum(sum(dldc(:,:,ii))); % Gradient of conv bias 1
                end
                
                dldb=dldc;
                
                % Through conv kernels 1
                dldk1_c=[];  % Gradient of conv filter 1
                dlda=zeros(size(a)); % Gradient of input to conv filter 1
                for rr=1:num_f1
                    dldk1_c{rr}=zeros(size(k1{rr}));
                    
                    for ii=1:size(b,1)
                        for jj=1:size(b,2)
                            
                            dldk1_c{rr} = dldk1_c{rr} + (a(ii:ii-1+k1_size(1), jj:jj-1+k1_size(2),:) * dldb(ii,jj,rr));
                            
                            dlda(ii:ii-1+k1_size(1), jj:jj-1+k1_size(2),:) =  dlda(ii:ii-1+k1_size(1), jj:jj-1+k1_size(2),:) + ... 
                                   k1{rr} *  dldb(ii,jj,rr);
                            
                        end
                    end
                end
                
                
                
                %%%%%%%%%%%%%%%%%%%% Update parameters %%%%%%%%%%%%%%%%%%%
                
                % Update conv layer 1 parameters                
                for jj=1:num_f1

                    % Update momentums and RMS terms and implement bias correction
                    V_k1{jj}=(beta1.*V_k1{jj}+(1-beta1).*dldk1_c{jj});   V_k1_cc{jj}=V_k1{jj}./(1-beta1^tot_ct);
                    S_k1{jj}=(beta2.*S_k1{jj}+(1-beta2).*(dldk1_c{jj}.^2));   S_k1_cc{jj}=S_k1{jj}./(1-beta2^tot_ct);
                    
                    V_b1{jj}=(beta1.*V_b1{jj}+(1-beta1).*dldb1_c{jj});    V_b1_cc{jj}=V_b1{jj}./(1-beta1^tot_ct);
                    S_b1{jj}=(beta2.*S_b1{jj}+(1-beta2).*(dldb1_c{jj}.^2));   S_b1_cc{jj}=S_b1{jj}./(1-beta2^tot_ct);
                    
                    % Then update parameters
                    k1{jj}=k1{jj}-(eta.*V_k1_cc{jj}./(sqrt(S_k1_cc{jj})+eps)); 
                    b1{jj}=b1{jj}-(eta.*V_b1_cc{jj}./(sqrt(S_b1_cc{jj})+eps)); 
                end
                
                % Update conv layer 2 parameters 
                for jj=1:num_f2
                    
                    V_k2{jj}=(beta1.*V_k2{jj}+(1-beta1).*dldk2_c{jj});  V_k2_cc{jj}=V_k2{jj}./(1-beta1^tot_ct);
                    S_k2{jj}=(beta2.*S_k2{jj}+(1-beta2).*(dldk2_c{jj}.^2));  S_k2_cc{jj}=S_k2{jj}./(1-beta2^tot_ct);
                    
                    V_b2{jj}=(beta1.*V_b2{jj}+(1-beta1).*dldb2_c{jj});   V_b2_cc{jj}=V_b2{jj}./(1-beta1^tot_ct);
                    S_b2{jj}=(beta2.*S_b2{jj}+(1-beta2).*(dldb2_c{jj}.^2));   S_b2_cc{jj}=S_b2{jj}./(1-beta2^tot_ct);
                    
                    k2{jj}=k2{jj}-(eta.*V_k2_cc{jj}./(sqrt(S_k2_cc{jj})+eps));
                    b2{jj}=b2{jj}-(eta.*V_b2_cc{jj}./(sqrt(S_b2_cc{jj})+eps));
                end
                
                % Update conv layer 3 parameters 
                for jj=1:num_f3
                    V_k3{jj}=(beta1.*V_k3{jj}+(1-beta1).*dldk3_c{jj});   V_k3_cc{jj}=V_k3{jj}./(1-beta1^tot_ct);
                    S_k3{jj}=(beta2.*S_k3{jj}+(1-beta2).*(dldk3_c{jj}.^2));  S_k3_cc{jj}=S_k3{jj}./(1-beta2^tot_ct);
                    
                    V_b3{jj}=(beta1.*V_b3{jj}+(1-beta1).*dldb3_c{jj});  V_b3_cc{jj}=V_b3{jj}./(1-beta1^tot_ct);
                    S_b3{jj}=(beta2.*S_b3{jj}+(1-beta2).*(dldb3_c{jj}.^2));  S_b3_cc{jj}=S_b3{jj}./(1-beta2^tot_ct);
                    
                    k3{jj}=k3{jj}-(eta.*V_k3_cc{jj}./(sqrt(S_k3_cc{jj})+eps));
                    b3{jj}=b3{jj}-(eta.*V_b3_cc{jj}./(sqrt(S_b3_cc{jj})+eps));
                end
                
                % Update fully connected layer 1 parameters
                V_w1_f=(beta1.*V_w1_f+(1-beta1).*dldw1);   V_w1_f_cc=V_w1_f./(1-beta1^tot_ct);
                S_w1_f=(beta2.*S_w1_f+(1-beta2).*(dldw1.^2));   S_w1_f_cc=S_w1_f./(1-beta2^tot_ct);
                V_b1_f=(beta1.*V_b1_f+(1-beta1).*dldb1);   V_b1_f_cc=V_b1_f./(1-beta1^tot_ct);
                S_b1_f=(beta2.*S_b1_f+(1-beta2).*(dldb1.^2));    S_b1_f_cc=S_b1_f./(1-beta2^tot_ct);
                
                w1_f=w1_f-(eta.*V_w1_f_cc./(sqrt(S_w1_f_cc)+eps));
                b1_f=b1_f-(eta.*V_b1_f_cc./(sqrt(S_b1_f_cc)+eps)); 
  
                % Update fully connected layer 2 parameters
                V_w2_f=(beta1.*V_w2_f+(1-beta1).*dldw2);      V_w2_f_cc= V_w2_f./(1-beta1^tot_ct);
                S_w2_f=(beta2.*S_w2_f+(1-beta2).*(dldw2.^2));   S_w2_f_cc= S_w2_f./(1-beta2^tot_ct);
                V_b2_f=(beta1.*V_b2_f+(1-beta1).*dldb2);     V_b2_f_cc=V_b2_f./(1-beta1^tot_ct);
                S_b2_f=(beta2.*S_b2_f+(1-beta2).*(dldb2.^2));    S_b2_f_cc=S_b2_f./(1-beta2^tot_ct);
                
                w2_f=w2_f-(eta.*V_w2_f_cc./(sqrt(S_w2_f_cc)+eps));
                b2_f=b2_f-(eta.*V_b2_f_cc./(sqrt(S_b2_f_cc)+eps));
                
                tot_ct=tot_ct+1;

            end
      
    end
    
    
    
        %%%%%%%%%%%%%%%%%%%%%%%%%% Validation Check %%%%%%%%%%%%%%%%%%%%%%%%%%

        % At the end of each epoch, perform a validation check to make sure you
        % aren't overfitting --> 
        for kk=1:2 % Get accuracy over both training and validation data
            if kk==1
                check_dat=train_dat;
                dataset='Train';
            else
                check_dat=val_dat;
                dataset='Val';
            end

            a_vec=[];
            for iiii=1:size(check_dat,1)

                IDX=iiii;
                temp=imread(check_dat{IDX,1}); % Read image from  source
                if length(size(temp))==3
                    X_initial=imresize(temp,[picsize, picsize]); % Resize image to be consistent dimensions for classification architecture
                    X=double(X_initial);

                    % Define ground truth vector in one-hot form
                    if (check_dat{IDX,2})==0
                       Y=[1; 0]; 
                    else
                       Y=[0; 1];
                    end
                    
                    % pad1
                    a=zeros(input_size(1)+4, input_size(2)+4, input_size(3),'double');
                    a(3:end-2,3:end-2,1:3)=X;

                    % conv1
                    b=[]; c=[];
                    for rr=1:num_f1 % Number of feature maps in first layer
                        for jj=1:size(a,2)-k1_size(1)+1 % Rows
                            for ii=1:size(a,1)-k1_size(2)+1 % Cols
                                kernel=reshape(k1{rr},1,[]);
                                temp=reshape(a(ii:ii+k1_size(1)-1,jj:jj+k1_size(2)-1,:),[],1);
                                b(ii,jj,rr)=kernel*temp;
                                c(ii,jj,rr)=b(ii,jj,rr)+b1{rr};
                            end
                        end
                    end

                    % relu1
                    logical_matrix1=(c>=0);
                    d=c.*logical_matrix1;

                    % pool1
                    e=[]; temp_ind1i=[]; temp_ind1j=[];
                    jj=1;
                    for jjj=0:2:size(d,2)-2
                        ii=1;
                        for iii=0:2:size(d,1)-2
                            temp=reshape(d(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f1);
                            [mx_temp, idx_temp] = max(temp);
                            temp_ind1j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj; % Store indices of max pool values for backprop
                            temp_ind1i(ii,jj,:)=iii+2-mod(idx_temp,2); 
                            e(ii,jj,:)=mx_temp;
                            ii=ii+1;
                        end
                        jj=jj+1;
                    end

                    % pad2
                    f=zeros(size(e,1)+4, size(e,2)+4, size(e,3),'double');
                    f(3:end-2, 3:end-2, 1:size(e,3))=e;

                    % conv2
                    g=[]; h=[];
                    for rr=1:num_f2 % Number of feature maps in first layer
                        for jj=1:size(f,2)-k2_size(1)+1 % Rows
                            for ii=1:size(f,1)-k2_size(2)+1 % Cols
                                kernel=reshape(k2{rr},1,[]);
                                temp=reshape(f(ii:ii+k2_size(1)-1,jj:jj+k2_size(2)-1,:),[],1);
                                g(ii,jj,rr)=kernel*temp;
                                h(ii,jj,rr)=g(ii,jj,rr)+b2{rr};
                            end
                        end
                    end

                    % relu2
                    logical_matrix2=(h>=0);
                    i=h.*logical_matrix2 ;

                    % pool2
                    j=[]; temp_ind2i=[]; temp_ind2j=[];
                    jj=1;
                    for jjj=0:2:size(i,2)-2
                        ii=1;
                        for iii=0:2:size(i,1)-2
                             temp=reshape(i(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f2); 
                            [mx_temp, idx_temp] = max(temp);
                            temp_ind2j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj;
                            temp_ind2i(ii,jj,:)=iii+2-mod(idx_temp,2);
                            j(ii,jj,:)=mx_temp;
                            ii=ii+1;
                        end
                        jj=jj+1;
                    end

                    % Pad3
                    k=zeros(size(j,1)+4, size(j,2)+4, size(j,3),'double');
                    k(3:end-2,3:end-2,:)=j;

                    % conv3
                    l=[]; m=[];
                    for rr=1:num_f3 % Number of feature maps
                        for jj=1:size(k,2)-k3_size(1)+1 % Rows
                            for ii=1:size(k,1)-k3_size(2)+1 % Cols
                                kernel=reshape(k3{rr},1,[]);
                                temp=reshape(k(ii:ii+k3_size(1)-1,jj:jj+k3_size(2)-1,:),[],1);
                                l(ii,jj,rr)=kernel*temp;
                                m(ii,jj,rr)=l(ii,jj,rr)+b3{rr};
                            end
                        end
                    end

                    % relu3
                    logical_matrix3=(m>=0);
                    n=m.*logical_matrix3;

                    % pool3
                    o=[]; temp_ind3i=[]; temp_ind3j=[];
                    jj=1;
                    for jjj=0:2:size(n,2)-2
                        ii=1;
                        for iii=0:2:size(n,1)-2
                            % Rearrange each feature map to column major vectors
                            temp=reshape(n(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f3);
                            [mx_temp, idx_temp] = max(temp);
                            temp_ind3j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj;
                            temp_ind3i(ii,jj,:)=iii+2-mod(idx_temp,2);
                            o(ii,jj,:)=mx_temp;
                            ii=ii+1;

                        end
                        jj=jj+1;
                    end

                    % Vectorize for fully connected layer
                    p=[]; fc=1;
                    for kk=1:size(o,3)
                        for jj=1:size(o,2)
                            for ii=1:size(o,1)
                                p(fc,1)=o(ii,jj,kk); fc=fc+1;
                            end
                        end
                    end


                    % FC Layer 1
                    q=w1_f*p; % Multiply by weight matrix
                    r=q+b1_f; % Add a bias
                    logical_matrix4=(r>=0); % relu
                    s=r.*logical_matrix4;

                    % FC Layer 2
                    t=w2_f*s; % Multiply by weight matrix
                    u=t+b2_f; % Add a bias
                    v=exp(u)./sum(exp(u)); % Nonlinear activation using softmax

                    if round(v(2))==Y(2)
                        acc=1;
                    else
                        acc=0;
                    end

                    a_vec=[a_vec; acc];

                end
            end

            % Display dataset accuracy
            disp(['Accuracy for ' dataset ' is ' num2str(mean(a_vec))])
            if strcmp(dataset,'Train')==1
                train_accuracy=[train_accuracy mean(a_vec)];
            end
            if  strcmp(dataset,'Val')==1
                val_accuracy=[val_accuracy mean(a_vec)];
            end

        end

        % End of epoch
    


end

%%%%%%%%%%%%%%%%%%%%% End of training %%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

check_dat=test_dat;
dataset='TEST';

a_vec=[]; p_vec=[];
for iiii=1:size(check_dat,1)

    IDX=iiii;
    temp=imread(check_dat{IDX,1}); % Read image from  source
    if length(size(temp))==3
        
        X_initial=imresize(temp,[picsize, picsize]); % Resize image to be consistent dimensions for classification architecture
        X=double(X_initial);

        % Define ground truth vector in one-hot form
        if (check_dat{IDX,2})==0
           Y=[1; 0]; 
        else
           Y=[0; 1];
        end

        % pad1
        a=zeros(input_size(1)+4, input_size(2)+4, input_size(3),'double');
        a(3:end-2,3:end-2,1:3)=X;

        % conv1
        b=[]; c=[];
        for rr=1:num_f1 % Number of feature maps in first layer
            for jj=1:size(a,2)-k1_size(1)+1 % Rows
                for ii=1:size(a,1)-k1_size(2)+1 % Cols
                    kernel=reshape(k1{rr},1,[]);
                    temp=reshape(a(ii:ii+k1_size(1)-1,jj:jj+k1_size(2)-1,:),[],1);
                    b(ii,jj,rr)=kernel*temp;
                    c(ii,jj,rr)=b(ii,jj,rr)+b1{rr};
                end
            end
        end

        % relu1
        logical_matrix1=(c>=0);
        d=c.*logical_matrix1;

        % pool1
        e=[]; temp_ind1i=[]; temp_ind1j=[];
        jj=1;
        for jjj=0:2:size(d,2)-2
            ii=1;
            for iii=0:2:size(d,1)-2
                temp=reshape(d(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f1);
                [mx_temp, idx_temp] = max(temp);
                temp_ind1j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj; % Store indices of max pool values for backprop
                temp_ind1i(ii,jj,:)=iii+2-mod(idx_temp,2); 
                e(ii,jj,:)=mx_temp;
                ii=ii+1;
            end
            jj=jj+1;
        end

        % pad2
        f=zeros(size(e,1)+4, size(e,2)+4, size(e,3),'double');
        f(3:end-2, 3:end-2, 1:size(e,3))=e;

        % conv2
        g=[]; h=[];
        for rr=1:num_f2 % Number of feature maps in first layer
            for jj=1:size(f,2)-k2_size(1)+1 % Rows
                for ii=1:size(f,1)-k2_size(2)+1 % Cols
                    kernel=reshape(k2{rr},1,[]);
                    temp=reshape(f(ii:ii+k2_size(1)-1,jj:jj+k2_size(2)-1,:),[],1);
                    g(ii,jj,rr)=kernel*temp;
                    h(ii,jj,rr)=g(ii,jj,rr)+b2{rr};
                end
            end
        end

        % relu2
        logical_matrix2=(h>=0);
        i=h.*logical_matrix2 ;

        % pool2
        j=[]; temp_ind2i=[]; temp_ind2j=[];
        jj=1;
        for jjj=0:2:size(i,2)-2
            ii=1;
            for iii=0:2:size(i,1)-2
                 temp=reshape(i(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f2); 
                [mx_temp, idx_temp] = max(temp);
                temp_ind2j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj;
                temp_ind2i(ii,jj,:)=iii+2-mod(idx_temp,2);
                j(ii,jj,:)=mx_temp;
                ii=ii+1;
            end
            jj=jj+1;
        end

        % Pad3
        k=zeros(size(j,1)+4, size(j,2)+4, size(j,3),'double');
        k(3:end-2,3:end-2,:)=j;

        % conv3
        l=[]; m=[];
        for rr=1:num_f3 % Number of feature maps
            for jj=1:size(k,2)-k3_size(1)+1 % Rows
                for ii=1:size(k,1)-k3_size(2)+1 % Cols
                    kernel=reshape(k3{rr},1,[]);
                    temp=reshape(k(ii:ii+k3_size(1)-1,jj:jj+k3_size(2)-1,:),[],1);
                    l(ii,jj,rr)=kernel*temp;
                    m(ii,jj,rr)=l(ii,jj,rr)+b3{rr};
                end
            end
        end

        % relu3
        logical_matrix3=(m>=0);
        n=m.*logical_matrix3;

        % pool3
        o=[]; temp_ind3i=[]; temp_ind3j=[];
        jj=1;
        for jjj=0:2:size(n,2)-2
            ii=1;
            for iii=0:2:size(n,1)-2
                % Rearrange each feature map to column major vectors
                temp=reshape(n(iii+1:iii+2,jjj+1:jjj+2,:),[],1,num_f3);
                [mx_temp, idx_temp] = max(temp);
                temp_ind3j(ii,jj,:)=1+floor((idx_temp-1)/2)+jjj;
                temp_ind3i(ii,jj,:)=iii+2-mod(idx_temp,2);
                o(ii,jj,:)=mx_temp;
                ii=ii+1;

            end
            jj=jj+1;
        end

        % Vectorize for fully connected layer
        p=[]; fc=1;
        for kk=1:size(o,3)
            for jj=1:size(o,2)
                for ii=1:size(o,1)
                    p(fc,1)=o(ii,jj,kk); fc=fc+1;
                end
            end
        end

        % FC Layer 1
        q=w1_f*p; % Multiply by weight matrix
        r=q+b1_f; % Add a bias
        logical_matrix4=(r>=0); % relu
        s=r.*logical_matrix4;

        % FC Layer 2
        t=w2_f*s; % Multiply by weight matrix
        u=t+b2_f; % Add a bias
        v=exp(u)./sum(exp(u)); % Nonlinear activation using softmax

        if round(v(2))==Y(2)
            acc=1;
        else
            acc=0;
        end

        a_vec= [a_vec; acc];
        p_vec(iiii)=round(v(2));

    end
end

disp(['Accuracy for ' dataset ' is ' num2str(mean(a_vec))])

figure(1)
plot(train_accuracy,'r')
hold on
plot(val_accuracy,'b')
title('Dataset Loss Per Epoch')
legend('Training Data','Validation Data')
            
            
            




  
  