%clear all
%close all

function train_dat_out=get_data()

    train_dat=[]; tc=1;
    
    disp('Retrieving spiderman images');

    % Get Spiderman images
    fileID = fopen([ pwd '\sm_http.txt']);
    C = textscan(fileID,'%s');
    fclose(fileID);

    tot_list=[];
    for i=1:length(C{1})
        p1='src="'; p2='"';
        temp=extractBetween(C{1}{i},p1,p2);
        tot_list=[tot_list; temp];
    end

    refined_list=[];
    ct=0;
    for kk=1:length(tot_list)
        try
           temp=imread(tot_list{kk});
           ct=ct+1;
           train_dat{tc,1}=tot_list{kk};
           train_dat{tc,2}=1;
           tc=tc+1;
        catch
        end
    end

    disp('Retrieving batman images');

    % Get batman images
    fileID = fopen([ pwd '\bm_html.txt']);
    C = textscan(fileID,'%s');
    fclose(fileID);

    tot_list=[];
    for i=1:length(C{1})
        p1='src="'; p2='"';
        temp=extractBetween(C{1}{i},p1,p2);
        tot_list=[tot_list; temp];
    end

    refined_list=[];
    ct=0;
    for kk=1:length(tot_list)
        try
           temp=imread(tot_list{kk});
           ct=ct+1;
           train_dat{tc,1}=tot_list{kk};
           train_dat{tc,2}=0;
           tc=tc+1;
        catch
        end
    end

    
    disp('Shuffling images')
    
    % Create shuffle function
    indices=randperm(size(train_dat,1));
    
    train_dat_out=[];
    for j=1:length(indices)
        train_dat_out{j,1}=train_dat{indices(j),1};
        train_dat_out{j,2}=train_dat{indices(j),2};
    end

end





