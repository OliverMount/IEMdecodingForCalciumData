

idx=9
tu=[]
for k=1:8
    sub_data=homo_data(find(dirIdx_homo==k),:,40:end);

    size(sub_data(:,idx,:))
    tu(k)=mean(sub_data(:,idx,:),'all')
    %tu(k)=mean(sub_data)
end
plot(tu)
[val,pos]=max(tu)