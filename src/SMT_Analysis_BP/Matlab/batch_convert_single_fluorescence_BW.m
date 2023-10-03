clear
a = which('Photobleaching_LL_EZ_fixed_01.vsi'); %give it an initial file to start with 
filelist = dir([fileparts(a) filesep 'Photobleaching_LL_EZ_fixed_*.vsi']);
fileNames = {filelist.name};
number_of_files=100 ; % pick the number of files to convert
% mkdir Movie
% mkdir gfp
for k=1:1:number_of_files
    rd=bfopen(fileNames{k});
    rd2=rd{1,1};
for i=1:length(rd2(:,1))

filename=sprintf('Photobleaching_LL_EZ_fixed_%d.tif',k);
imwrite(rd2{i,1},filename,'WriteMode','append');
end

end