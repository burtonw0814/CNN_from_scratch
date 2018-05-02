function train_dat_out=shuffle_data(train_dat_in)

    indices=randperm(size(train_dat_in,1));
    
    train_dat_out=[];
    for j=1:length(indices)
        train_dat_out{j,1}=train_dat_in{indices(j),1};
        train_dat_out{j,2}=train_dat_in{indices(j),2};
    end

end